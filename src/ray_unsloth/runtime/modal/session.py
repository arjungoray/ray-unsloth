"""Modal-backed GPU runtime with local Ray orchestration."""

from __future__ import annotations

import atexit
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from ray_unsloth.config import LoRAConfig, ModelConfig, RuntimeConfig
from ray_unsloth.errors import RayUnavailableError

_ENGINE_REGISTRY: dict[tuple[str, str], Any] = {}


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
        if key not in {"volume_name", "volume_mount_path"}
    }


def _modal_invoke_impl(
    actor_kind: str,
    session_id: str,
    init_kwargs: dict[str, Any],
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run one actor method inside a Modal GPU container."""

    key = (actor_kind, session_id)
    actor = _ENGINE_REGISTRY.get(key)
    if actor is None:
        _sync_modal_volume(init_kwargs, "reload")
        if actor_kind == "trainer":
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


class _ModalMethod:
    def __init__(self, actor: "ModalActorHandle", method_name: str):
        self._actor = actor
        self._method_name = method_name

    def remote(self, *args, **kwargs):
        return self._actor._session.invoke(
            actor_kind=self._actor.actor_kind,
            session_id=self._actor.session_id,
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
        init_kwargs: dict[str, Any],
    ) -> None:
        self._session = session
        self.actor_kind = actor_kind
        self.session_id = session_id
        self.init_kwargs = init_kwargs

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
        self._app, self._invoke = self._build_modal_app()
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
        packages = [
            "ray>=2.55.1",
            "pyyaml>=6.0.3",
            "torch>=2.10.0,<2.11.0",
            "transformers>=5.5.0,<5.5.1",
            "bitsandbytes>=0.49.2",
            "unsloth>=2026.4.8",
            "unsloth-zoo>=2026.4.9",
        ]
        if self.config.model.fast_inference:
            packages.append("vllm>=0.20.0")
        image = modal.Image.debian_slim(python_version=modal_config.python_version).pip_install(*packages)
        source_root = Path(__file__).resolve().parents[3]
        package_dir = source_root / "ray_unsloth"
        if hasattr(image, "add_local_dir"):
            image = image.add_local_dir(
                str(package_dir),
                remote_path="/root/ray_unsloth_src/ray_unsloth",
                copy=True,
            )
        if hasattr(image, "env"):
            image = image.env({"PYTHONPATH": "/root/ray_unsloth_src"})

        volume = modal.Volume.from_name(modal_config.volume_name, create_if_missing=True)
        function_kwargs = {
            "image": image,
            "gpu": modal_config.gpu,
            "timeout": modal_config.timeout,
            "scaledown_window": modal_config.scaledown_window,
            "volumes": {modal_config.volume_mount_path: volume},
        }
        return app, app.function(**function_kwargs)(_modal_invoke_impl)

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
        init_kwargs: dict[str, Any],
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return self._invoke.remote(actor_kind, session_id, init_kwargs, method_name, args, kwargs)

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
            "lora_config": self._lora_config(lora_rank, seed=seed, target_modules=target_modules),
            "checkpoint_root": self.config.checkpoint_root,
            "model_path": model_path,
            "with_optimizer": with_optimizer,
            "metadata": metadata or {},
            "volume_name": self.config.modal.volume_name,
            "volume_mount_path": self.config.modal.volume_mount_path,
        }
        actor = ModalActorHandle(
            session=self,
            actor_kind="trainer",
            session_id=session_id,
            init_kwargs=init_kwargs,
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
        for _ in range(replica_count):
            init_kwargs = {
                "session_id": session_id,
                "model_config": self._model_config(base_model),
                "lora_config": self._lora_config(None),
                "checkpoint_root": self.config.checkpoint_root,
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
                )
            )
        self.sampler_actors[session_id] = actors
        return session_id, actors

    def _model_config(self, base_model: str | None) -> ModelConfig:
        if base_model is None:
            return self.config.model
        return replace(self.config.model, base_model=base_model)

    def _lora_config(
        self,
        rank: int | None,
        *,
        seed: int | None = None,
        target_modules: list[str] | None = None,
    ) -> LoRAConfig:
        updates: dict[str, Any] = {}
        if rank is not None:
            updates["rank"] = rank
        if seed is not None:
            updates["random_state"] = seed
        if target_modules is not None:
            updates["target_modules"] = target_modules
        if not updates:
            return self.config.lora
        return replace(self.config.lora, **updates)
