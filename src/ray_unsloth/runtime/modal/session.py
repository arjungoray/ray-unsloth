"""Modal-backed GPU runtime with local Ray orchestration."""

from __future__ import annotations

import atexit
import os
import threading
import uuid
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from ray_unsloth.config import LoRAConfig, ModelConfig, RuntimeConfig
from ray_unsloth.errors import RayUnavailableError

_ENGINE_REGISTRY: dict[tuple[str, str, int], Any] = {}
_TRAINER_INVOCATION_LOCK = threading.RLock()
VLLM_CU130_WHEEL = (
    "https://github.com/vllm-project/vllm/releases/download/v0.19.1/"
    "vllm-0.19.1+cu130-cp38-abi3-manylinux_2_35_x86_64.whl"
)
FLASH_ATTN_VERSION = "2.8.3"
FLASH_ATTN_TORCH_VERSION = "2.8"
FLASH_ATTN_WHEEL_BASE_URL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3"
FLASH_LINEAR_ATTENTION_PACKAGE = "flash-linear-attention==0.5.0"
CAUSAL_CONV1D_VERSION = "1.6.1"
CAUSAL_CONV1D_TAG = "v1.6.1.post4"
CAUSAL_CONV1D_WHEEL_BASE_URL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4"
)


def _sync_modal_volume(init_kwargs: dict[str, Any], method_name: str) -> None:
    volume_name = init_kwargs.get("volume_name")
    if not volume_name:
        return
    try:
        import modal

        method = getattr(modal.Volume.from_name(volume_name), method_name, None)
        if callable(method):
            method()
    except Exception:
        # Volume sync is a best-effort bridge between sequential trainer and
        # sampler calls. The underlying method errors will still surface.
        return


def _actor_kwargs(init_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in init_kwargs.items()
        if key not in {"volume_name", "volume_mount_path", "distributed_config"}
    }


def _visible_gpu_count() -> int | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([device for device in visible.split(",") if device.strip()])
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return None


def _init_modal_local_ray(init_kwargs: dict[str, Any]):
    try:
        import ray
    except ImportError as exc:
        raise RayUnavailableError("Ray is required for Modal DDP orchestration.") from exc
    if not ray.is_initialized():
        ray.init(
            namespace="ray-unsloth-modal",
            ignore_reinit_error=True,
        )
    return ray


def _create_modal_distributed_trainer(init_kwargs: dict[str, Any]):
    from ray_unsloth.runtime.ray.distributed_trainer import (
        DistributedTrainerCoordinator,
        DistributedTrainerWorkerActor,
    )

    distributed = init_kwargs["distributed_config"]
    visible_count = _visible_gpu_count()
    if visible_count is not None and visible_count < distributed.gpus_per_node:
        raise RuntimeError(
            "Modal DDP requested "
            f"{distributed.gpus_per_node} GPU(s), but only {visible_count} are visible in the container."
        )
    ray = _init_modal_local_ray(init_kwargs)
    workers = []
    for rank in range(distributed.gpus_per_node):
        workers.append(
            DistributedTrainerWorkerActor.options(
                num_gpus=1,
                num_cpus=1,
            ).remote(
                session_id=init_kwargs["session_id"],
                model_config=init_kwargs["model_config"],
                lora_config=init_kwargs["lora_config"],
                checkpoint_root=init_kwargs["checkpoint_root"],
                speed_config=init_kwargs.get("speed_config"),
                rank=rank,
                world_size=distributed.gpus_per_node,
                backend=distributed.backend,
                model_path=init_kwargs.get("model_path"),
                with_optimizer=bool(init_kwargs.get("with_optimizer", False)),
                metadata=init_kwargs.get("metadata") or {},
            )
        )
    init_method = ray.get(workers[0].get_process_group_endpoint.remote())
    ray.get([worker.initialize.remote(init_method) for worker in workers])
    return DistributedTrainerCoordinator(workers=workers, world_size=distributed.gpus_per_node)


def _modal_function_kwargs(config: RuntimeConfig, image, volume) -> dict[str, Any]:
    modal_config = config.modal
    kwargs = {
        "image": image,
        "gpu": modal_config.gpu,
        "timeout": modal_config.timeout,
        "scaledown_window": modal_config.scaledown_window,
        "volumes": {modal_config.volume_mount_path: volume},
        # The runtime stores actors in container memory. Keep one warm container
        # so training, sampling, save, and optimizer calls hit the same actor.
        "max_containers": 1,
    }
    if modal_config.max_inputs is not None:
        kwargs["max_inputs"] = modal_config.max_inputs
    return kwargs


def _modal_wants_vllm(model_config: ModelConfig) -> bool:
    if _modal_requires_qwen3_5_transformers(model_config):
        return False
    if model_config.fast_inference is True:
        return True
    return model_config.fast_inference == "auto" and not model_config.trust_remote_code


def _modal_any_wants_vllm(config: RuntimeConfig) -> bool:
    return _modal_wants_vllm(config.model) or any(
        _modal_wants_vllm(model_config.model) for model_config in config.model_configs.values()
    )


def _modal_wants_flash_attention_package(config: RuntimeConfig) -> bool:
    return not _modal_any_wants_vllm(config)


def _modal_torch_cuda_tags(config: RuntimeConfig) -> tuple[str, str]:
    if _modal_wants_flash_attention_package(config):
        return "2.8", "cu12"
    return "2.10", "cu13"


def _modal_requires_qwen3_5_transformers(model_config: ModelConfig) -> bool:
    normalized = model_config.base_model.lower().replace("-", "_").replace("/", "_").replace(".", "_")
    return "qwen3_5" in normalized


def _modal_any_requires_qwen3_5_transformers(config: RuntimeConfig) -> bool:
    return _modal_requires_qwen3_5_transformers(config.model) or any(
        _modal_requires_qwen3_5_transformers(model_config.model) for model_config in config.model_configs.values()
    )


def _modal_transformers_package(config: RuntimeConfig) -> str:
    if _modal_any_requires_qwen3_5_transformers(config):
        return "transformers==5.5.0"
    if _modal_any_wants_vllm(config):
        return "transformers==4.57.6"
    return "transformers==5.5.0"


def _modal_huggingface_hub_package(config: RuntimeConfig) -> str:
    if _modal_transformers_package(config) == "transformers==5.5.0":
        return "huggingface_hub==1.14.0"
    return "huggingface_hub==0.36.2"


def _modal_torch_backend_packages(config: RuntimeConfig) -> list[str]:
    if _modal_wants_flash_attention_package(config):
        packages = [
            "torch==2.8.0",
            "torchvision==0.23.0",
            "xformers==0.0.32.post2",
            "packaging==26.2",
            "wheel==0.45.1",
        ]
    else:
        packages = [
            "torch==2.10.0",
            "torchvision==0.25.0",
            "xformers==0.0.34",
            "packaging==26.2",
            "wheel==0.45.1",
        ]
    if _modal_any_wants_vllm(config):
        packages.append(VLLM_CU130_WHEEL)
    return packages


def _modal_flash_attention_packages(config: RuntimeConfig) -> list[str]:
    if not _modal_wants_flash_attention_package(config):
        return []
    python_tag = config.modal.python_version.replace(".", "")
    wheel_name = (
        f"flash_attn-{FLASH_ATTN_VERSION}+cu12torch{FLASH_ATTN_TORCH_VERSION}"
        f"cxx11abiTRUE-cp{python_tag}-cp{python_tag}-linux_x86_64.whl"
    )
    wheel_url = f"{FLASH_ATTN_WHEEL_BASE_URL}/{wheel_name.replace('+', '%2B')}"
    return [f"flash-attn @ {wheel_url}"]


def _modal_causal_conv1d_package(config: RuntimeConfig) -> str:
    python_tag = config.modal.python_version.replace(".", "")
    torch_tag, cuda_tag = _modal_torch_cuda_tags(config)
    wheel_name = (
        f"causal_conv1d-{CAUSAL_CONV1D_VERSION}+{cuda_tag}torch{torch_tag}"
        f"cxx11abiTRUE-cp{python_tag}-cp{python_tag}-linux_x86_64.whl"
    )
    wheel_url = f"{CAUSAL_CONV1D_WHEEL_BASE_URL}/{wheel_name.replace('+', '%2B')}"
    return f"causal-conv1d @ {wheel_url}"


def _modal_linear_attention_packages(config: RuntimeConfig) -> list[str]:
    return [
        FLASH_LINEAR_ATTENTION_PACKAGE,
        _modal_causal_conv1d_package(config),
    ]


def _modal_base_image(modal, config: RuntimeConfig, flash_attention_packages: list[str]):
    del flash_attention_packages
    modal_config = config.modal
    return modal.Image.debian_slim(python_version=modal_config.python_version)


def _modal_python_packages(config: RuntimeConfig) -> list[str]:
    return [
        "accelerate==1.13.0",
        "bitsandbytes==0.49.2",
        "hf-transfer==0.1.9",
        _modal_huggingface_hub_package(config),
        "peft==0.19.1",
        "ray==2.55.1",
        "pyyaml==6.0.3",
        _modal_transformers_package(config),
        "trl==0.24.0",
        "unsloth[colab-new]>=2026.5.2",
        "unsloth-zoo>=2026.5.1",
    ]


def _modal_invoke_impl(
    actor_kind: str,
    session_id: str,
    init_kwargs: dict[str, Any],
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    replica_index: int = 0,
) -> Any:
    """Run one actor method inside a Modal GPU container."""

    lock = _TRAINER_INVOCATION_LOCK if actor_kind == "trainer" else nullcontext()
    with lock:
        key = (actor_kind, session_id, replica_index)
        actor = _ENGINE_REGISTRY.get(key)
        if actor is None:
            _sync_modal_volume(init_kwargs, "reload")
            distributed_config = init_kwargs.get("distributed_config")
            if actor_kind == "trainer" and getattr(distributed_config, "enabled", False):
                actor = _create_modal_distributed_trainer(init_kwargs)
            elif actor_kind == "trainer":
                from ray_unsloth.runtime.ray.trainer_actor import TrainerActorImpl

                actor = TrainerActorImpl(**_actor_kwargs(init_kwargs))
            elif actor_kind == "sampler":
                from ray_unsloth.runtime.ray.sampler_actor import SamplerActorImpl

                actor = SamplerActorImpl(**_actor_kwargs(init_kwargs))
            else:
                raise ValueError(f"Unsupported Modal actor kind: {actor_kind}")
            _ENGINE_REGISTRY[key] = actor

        result = getattr(actor, method_name)(*args, **kwargs)
        if method_name.startswith("save_"):
            _sync_modal_volume(init_kwargs, "commit")
        return result


def _modal_actor_service_invoke(self, session_id, init_kwargs, method_name, args, kwargs):
    return _modal_invoke_impl(
        self.actor_kind,
        session_id,
        init_kwargs,
        method_name,
        args,
        kwargs,
        replica_index=self.replica_index,
    )


try:
    import modal as _modal
except ImportError:  # pragma: no cover - modal is optional until runtime use
    _modal = None

if _modal is not None:

    class ModalActorService:
        actor_kind = _modal.parameter()
        pool_id = _modal.parameter()
        replica_index = _modal.parameter(default=0)

        @_modal.method()
        def invoke(self, session_id, init_kwargs, method_name, args, kwargs):
            return _modal_actor_service_invoke(self, session_id, init_kwargs, method_name, args, kwargs)

    ModalActorService.__annotations__ = {
        "actor_kind": str,
        "pool_id": str,
        "replica_index": int,
    }


def _modal_actor_service_class(modal):
    actor_cls = globals().get("ModalActorService")
    if actor_cls is not None:
        return actor_cls
    actor_cls = type(
        "ModalActorService",
        (),
        {
            "__module__": __name__,
            "__qualname__": "ModalActorService",
            "__annotations__": {
                "actor_kind": str,
                "pool_id": str,
                "replica_index": int,
            },
            "actor_kind": modal.parameter(),
            "pool_id": modal.parameter(),
            "replica_index": modal.parameter(default=0),
            "invoke": modal.method()(_modal_actor_service_invoke),
        },
    )
    globals()["ModalActorService"] = actor_cls
    return actor_cls


class _ModalMethod:
    def __init__(self, actor: "ModalActorHandle", method_name: str):
        self._actor = actor
        self._method_name = method_name

    def remote(self, *args, **kwargs):
        return self._actor._session.invoke(
            actor_kind=self._actor.actor_kind,
            session_id=self._actor.session_id,
            pool_id=self._actor.pool_id,
            replica_index=self._actor.replica_index,
            init_kwargs=self._actor.init_kwargs,
            method_name=self._method_name,
            args=args,
            kwargs=kwargs,
        )

    async def remote_async(self, *args, **kwargs):
        return await self._actor._session.invoke_async(
            actor_kind=self._actor.actor_kind,
            session_id=self._actor.session_id,
            pool_id=self._actor.pool_id,
            replica_index=self._actor.replica_index,
            init_kwargs=self._actor.init_kwargs,
            method_name=self._method_name,
            args=args,
            kwargs=kwargs,
        )


class ModalActorHandle:
    """Small local handle exposing Ray-like actor method calls."""

    def __init__(
        self,
        *,
        session: "ModalSession",
        actor_kind: str,
        session_id: str,
        pool_id: str | None = None,
        init_kwargs: dict[str, Any],
        replica_index: int = 0,
    ) -> None:
        self._session = session
        self.actor_kind = actor_kind
        self.session_id = session_id
        self.pool_id = pool_id or session_id
        self.init_kwargs = init_kwargs
        self.replica_index = replica_index

    def get_tokenizer(self):
        """Load a plain local tokenizer instead of deserializing Unsloth state."""

        from transformers import AutoTokenizer

        model_config = self.init_kwargs["model_config"]
        return AutoTokenizer.from_pretrained(
            model_config.base_model,
            trust_remote_code=model_config.trust_remote_code,
        )

    def __getattr__(self, method_name: str) -> _ModalMethod:
        return _ModalMethod(self, method_name)


class ModalSession:
    """Creates Modal GPU actor handles while keeping Ray local."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.training_actors: dict[str, Any] = {}
        self.sampler_actors: dict[str, list[Any]] = {}
        self._ray = self._init_local_ray()
        self._app, self._actor_cls = self._build_modal_app()
        self._output_context = None
        self._runner = None
        self._start_app()

    def _init_local_ray(self):
        try:
            import ray
        except ImportError as exc:
            raise RayUnavailableError("Ray is required for local orchestration.") from exc
        if not ray.is_initialized():
            ray.init(
                address=self.config.ray.address,
                namespace=self.config.ray.namespace,
                ignore_reinit_error=self.config.ray.ignore_reinit_error,
            )
        return ray

    def _build_modal_app(self):
        try:
            import modal
        except ImportError as exc:
            raise RuntimeError(
                "Modal is required when modal.enabled is true. Install with "
                "`pip install -e '.[modal]'`."
            ) from exc

        modal_config = self.config.modal
        app = modal.App(modal_config.app_name)
        packages = _modal_python_packages(self.config)
        torch_backend_packages = _modal_torch_backend_packages(self.config)
        flash_attention_packages = _modal_flash_attention_packages(self.config)
        linear_attention_packages = _modal_linear_attention_packages(self.config)
        image = (
            _modal_base_image(modal, self.config, flash_attention_packages)
            .apt_install("git", "ninja-build", "build-essential")
            .uv_pip_install(*torch_backend_packages, extra_options="--torch-backend=auto", gpu=modal_config.gpu)
            .env({"MAX_JOBS": "4"})
        )
        if flash_attention_packages:
            image = image.uv_pip_install(*flash_attention_packages)
        if linear_attention_packages:
            image = image.uv_pip_install(*linear_attention_packages)
        image = (
            image
            .uv_pip_install(*packages)
            .env({"HF_HOME": "/model_cache", "PYTHONPATH": "/root/ray_unsloth_src"})
        )
        source_root = Path(__file__).resolve().parents[3]
        package_dir = source_root / "ray_unsloth"
        if hasattr(image, "add_local_dir"):
            image = image.add_local_dir(
                str(package_dir),
                remote_path="/root/ray_unsloth_src/ray_unsloth",
                copy=True,
            )
        if hasattr(image, "env"):
            image = image.env({"HF_HOME": "/model_cache", "PYTHONPATH": "/root/ray_unsloth_src"})

        volume = modal.Volume.from_name(modal_config.volume_name, create_if_missing=True)
        function_kwargs = _modal_function_kwargs(self.config, image, volume)
        max_inputs = function_kwargs.pop("max_inputs", None)
        actor_cls = app.cls(**function_kwargs)(_modal_actor_service_class(modal))
        if max_inputs is not None:
            actor_cls = actor_cls.with_concurrency(max_inputs=max_inputs)
        return app, actor_cls

    def _start_app(self) -> None:
        import modal

        output_context = modal.enable_output()
        output_context.__enter__()
        self._output_context = output_context
        runner = self._app.run()
        self._runner = runner
        runner.__enter__()
        atexit.register(self.close)

    def close(self) -> None:
        runner = self._runner
        output_context = self._output_context
        self._runner = None
        self._output_context = None
        if runner is not None:
            runner.__exit__(None, None, None)
        if output_context is not None:
            output_context.__exit__(None, None, None)

    def invoke(
        self,
        *,
        actor_kind: str,
        session_id: str,
        pool_id: str,
        replica_index: int,
        init_kwargs: dict[str, Any],
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        actor = self._actor_cls(
            actor_kind=actor_kind,
            pool_id=pool_id,
            replica_index=replica_index,
        )
        return actor.invoke.remote(session_id, init_kwargs, method_name, args, kwargs)

    async def invoke_async(
        self,
        *,
        actor_kind: str,
        session_id: str,
        pool_id: str,
        replica_index: int,
        init_kwargs: dict[str, Any],
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        actor = self._actor_cls(
            actor_kind=actor_kind,
            pool_id=pool_id,
            replica_index=replica_index,
        )
        return await actor.invoke.remote.aio(session_id, init_kwargs, method_name, args, kwargs)

    def create_training_actor(
        self,
        *,
        base_model: str | None = None,
        lora_rank: int | None = None,
        seed: int | None = None,
        target_modules: list[str] | None = None,
        model_path: str | None = None,
        with_optimizer: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, Any]:
        session_id = f"train-{uuid.uuid4().hex}"
        init_kwargs = {
            "session_id": session_id,
            "model_config": self._model_config(base_model),
            "lora_config": self._lora_config(
                lora_rank,
                base_model=base_model,
                seed=seed,
                target_modules=target_modules,
            ),
            "checkpoint_root": self.config.checkpoint_root,
            "speed_config": self.config.speed,
            "model_path": model_path,
            "with_optimizer": with_optimizer,
            "metadata": metadata or {},
            "distributed_config": self.config.distributed,
            "volume_name": self.config.modal.volume_name,
            "volume_mount_path": self.config.modal.volume_mount_path,
        }
        actor = ModalActorHandle(
            session=self,
            actor_kind="trainer",
            session_id=session_id,
            pool_id=self.config.modal.trainer_pool_key or session_id,
            init_kwargs=init_kwargs,
            replica_index=0,
        )
        self.training_actors[session_id] = actor
        return session_id, actor

    def create_sampler_actors(
        self,
        *,
        base_model: str | None = None,
        model_path: str | None = None,
        replicas: int | None = None,
    ) -> tuple[str, list[Any]]:
        session_id = f"sample-{uuid.uuid4().hex}"
        replica_count = replicas if replicas is not None else self.config.resources.sampler_replicas
        actors = []
        for index in range(replica_count):
            init_kwargs = {
                "session_id": session_id,
                "model_config": self._model_config(base_model),
                "lora_config": self._lora_config(None, base_model=base_model),
                "checkpoint_root": self.config.checkpoint_root,
                "speed_config": self.config.speed,
                "model_path": model_path,
                "volume_name": self.config.modal.volume_name,
                "volume_mount_path": self.config.modal.volume_mount_path,
            }
            actors.append(
                ModalActorHandle(
                    session=self,
                    actor_kind="sampler",
                    session_id=session_id,
                    init_kwargs=init_kwargs,
                    replica_index=index,
                )
            )
        self.sampler_actors[session_id] = actors
        return session_id, actors

    def _model_config(self, base_model: str | None) -> ModelConfig:
        model_config, _lora_config = self.config.resolve_model_configs(base_model)
        return model_config

    def _lora_config(
        self,
        rank: int | None,
        *,
        base_model: str | None = None,
        seed: int | None = None,
        target_modules: list[str] | None = None,
    ) -> LoRAConfig:
        _model_config, lora_config = self.config.resolve_model_configs(base_model)
        updates: dict[str, Any] = {}
        if rank is not None:
            updates["rank"] = rank
        if seed is not None:
            updates["random_state"] = seed
        if target_modules is not None:
            updates["target_modules"] = target_modules
        if not updates:
            return lora_config
        return replace(lora_config, **updates)
