from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Any

from ray_unsloth.errors import RayUnavailableError

_ENGINE_REGISTRY: dict[tuple[str, str, int], Any] = {}
_TRAINER_INVOCATION_LOCK = threading.RLock()


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
    import os

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
        raise RayUnavailableError(
            "Ray is required for Modal DDP orchestration.",
            code="RU-3004",
            hint="Install ray or disable distributed Modal training.",
        ) from exc
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
else:
    ModalActorService = None


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
    def __init__(self, actor, method_name: str):
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
        session,
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


__all__ = [
    "_ENGINE_REGISTRY",
    "_TRAINER_INVOCATION_LOCK",
    "ModalActorHandle",
    "ModalActorService",
    "_ModalMethod",
    "_actor_kwargs",
    "_create_modal_distributed_trainer",
    "_init_modal_local_ray",
    "_modal_actor_service_class",
    "_modal_actor_service_invoke",
    "_modal_invoke_impl",
    "_sync_modal_volume",
    "_visible_gpu_count",
]
