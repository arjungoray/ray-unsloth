"""Modal-backed GPU runtime with local Ray orchestration."""

from __future__ import annotations

import atexit
import uuid
from pathlib import Path
from typing import Any

from ray_unsloth.config import RuntimeConfig
from ray_unsloth.errors import RayUnavailableError
from ray_unsloth.runtime._resolve import resolve_actor_configs
from ray_unsloth.runtime.modal.downloads import (
    DOWNLOAD_APP_SUFFIX,
    _ensure_download_endpoint,
)
from ray_unsloth.runtime.modal.handles import (
    _ENGINE_REGISTRY,
    _TRAINER_INVOCATION_LOCK,
    ModalActorHandle,
    _create_modal_distributed_trainer,
    _modal_actor_service_class,
)
from ray_unsloth.runtime.modal.image import (
    VLLM_CU130_WHEEL as _VLLM_CU130_WHEEL,
)
from ray_unsloth.runtime.modal.image import (
    _modal_base_image,
    _modal_flash_attention_packages,
    _modal_linear_attention_packages,
    _modal_python_packages,
    _modal_torch_backend_packages,
)

from . import handles as _handles_module

VLLM_CU130_WHEEL = _VLLM_CU130_WHEEL


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


def _modal_invoke_impl(
    actor_kind: str,
    session_id: str,
    init_kwargs: dict[str, Any],
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    replica_index: int = 0,
) -> Any:
    _handles_module._create_modal_distributed_trainer = _create_modal_distributed_trainer
    _handles_module._ENGINE_REGISTRY = _ENGINE_REGISTRY
    _handles_module._TRAINER_INVOCATION_LOCK = _TRAINER_INVOCATION_LOCK
    return _handles_module._modal_invoke_impl(
        actor_kind,
        session_id,
        init_kwargs,
        method_name,
        args,
        kwargs,
        replica_index=replica_index,
    )


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
            raise RayUnavailableError(
                "Modal is required when modal.enabled is true. Install with `pip install -e '.[modal]'`."
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
        image = image.uv_pip_install(*packages).env({"HF_HOME": "/model_cache", "PYTHONPATH": "/root/ray_unsloth_src"})
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

        self._download_fn = _ensure_download_endpoint(
            modal=modal,
            app_name=f"{modal_config.app_name}{DOWNLOAD_APP_SUFFIX}",
            volume_name=modal_config.volume_name,
            mount_path=modal_config.volume_mount_path,
            timeout=modal_config.timeout,
            python_version=modal_config.python_version,
        )
        return app, actor_cls

    def build_sampler_download_url(self, archive_relpath: str, token: str, expires_at: int) -> str | None:
        fn = getattr(self, "_download_fn", None)
        if fn is None:
            return None
        try:
            base = fn.get_web_url()
        except Exception:
            return None
        if not base:
            return None
        from urllib.parse import urlencode

        query = urlencode({"archive": archive_relpath, "expires": expires_at, "token": token})
        return f"{base}?{query}"

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
        model_config, lora_config = resolve_actor_configs(
            self.config,
            base_model=base_model,
            lora_rank=lora_rank,
            seed=seed,
            target_modules=target_modules,
        )
        init_kwargs = {
            "session_id": session_id,
            "model_config": model_config,
            "lora_config": lora_config,
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
            model_config, lora_config = resolve_actor_configs(self.config, base_model=base_model)
            init_kwargs = {
                "session_id": session_id,
                "model_config": model_config,
                "lora_config": lora_config,
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


__all__ = [
    "DOWNLOAD_APP_SUFFIX",
    "VLLM_CU130_WHEEL",
    "_ENGINE_REGISTRY",
    "_TRAINER_INVOCATION_LOCK",
    "ModalActorHandle",
    "ModalSession",
    "_create_modal_distributed_trainer",
    "_ensure_download_endpoint",
    "_modal_base_image",
    "_modal_flash_attention_packages",
    "_modal_function_kwargs",
    "_modal_invoke_impl",
    "_modal_linear_attention_packages",
    "_modal_python_packages",
    "_modal_torch_backend_packages",
]
