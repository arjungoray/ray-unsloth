"""Main entry point for Ray Unsloth sessions."""

from __future__ import annotations

from typing import Any

from ray_unsloth.clients.sampling import SamplingClient
from ray_unsloth.clients.training import TrainingClient
from ray_unsloth.config import RuntimeConfig, load_config
from ray_unsloth.runtime.modal import ModalSession
from ray_unsloth.runtime.ray import RaySession
from ray_unsloth.types import GetServerCapabilitiesResponse


class ServiceClient:
    """Main control-plane handle, matching Tinker's client-construction layer."""

    def __init__(self, config: str | RuntimeConfig | dict[str, Any] | None = None):
        self.config = load_config(config)
        self._session = ModalSession(self.config) if self.config.modal.enabled else RaySession(self.config)

    def get_server_capabilities(self) -> GetServerCapabilitiesResponse:
        supported = self.config.supported_models or [self.config.model.base_model]
        return GetServerCapabilitiesResponse(
            supported_models=supported,
            max_sampler_replicas=self.config.resources.sampler_replicas,
            features={
                "losses": ["cross_entropy"],
                "checkpointing": True,
                "ray_namespace": self.config.ray.namespace,
                "runtime_backend": "modal" if self.config.modal.enabled else "ray",
            },
        )

    def create_lora_training_client(
        self,
        *,
        base_model: str | None = None,
        rank: int | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> TrainingClient:
        # Accept extra Tinker-style keyword fields that are represented by config
        # in this MVP, while keeping the call source-compatible.
        del kwargs
        session_id, actor = self._session.create_training_actor(
            base_model=base_model,
            lora_rank=rank,
            metadata=metadata,
        )
        return TrainingClient(session_id=session_id, actor=actor, service=self)

    def create_sampling_client(
        self,
        *,
        base_model: str | None = None,
        model_path: str | None = None,
        replicas: int | None = None,
        **kwargs,
    ) -> SamplingClient:
        del kwargs
        session_id, actors = self._session.create_sampler_actors(
            base_model=base_model,
            model_path=model_path,
            replicas=replicas,
        )
        return SamplingClient(session_id=session_id, actors=actors)

    def create_training_client_from_state(self, path: str, **kwargs) -> TrainingClient:
        base_model = kwargs.pop("base_model", None)
        rank = kwargs.pop("rank", None)
        del kwargs
        session_id, actor = self._session.create_training_actor(
            base_model=base_model,
            lora_rank=rank,
            model_path=path,
            with_optimizer=False,
        )
        return TrainingClient(session_id=session_id, actor=actor, service=self)

    def create_training_client_from_state_with_optimizer(self, path: str, **kwargs) -> TrainingClient:
        base_model = kwargs.pop("base_model", None)
        rank = kwargs.pop("rank", None)
        del kwargs
        session_id, actor = self._session.create_training_actor(
            base_model=base_model,
            lora_rank=rank,
            model_path=path,
            with_optimizer=True,
        )
        return TrainingClient(session_id=session_id, actor=actor, service=self)
