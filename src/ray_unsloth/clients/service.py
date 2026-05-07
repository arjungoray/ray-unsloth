"""Main entry point for Ray Unsloth sessions."""

from __future__ import annotations

from typing import Any

from ray_unsloth.clients.sampling import SamplingClient
from ray_unsloth.clients.training import TrainingClient
from ray_unsloth.config import RuntimeConfig, load_config, lora_target_modules_for_flags
from ray_unsloth.runtime.modal import ModalSession
from ray_unsloth.runtime.ray import RaySession
from ray_unsloth.types import GetServerCapabilitiesResponse


class ServiceClient:
    """Main control-plane handle, matching Tinker's client-construction layer."""

    def __init__(
        self,
        user_metadata: dict[str, str] | str | RuntimeConfig | dict[str, Any] | None = None,
        project_id: str | None = None,
        config: str | RuntimeConfig | dict[str, Any] | None = None,
        base_url: str | None = None,
    ):
        del base_url
        if config is None and self._looks_like_config(user_metadata):
            config = user_metadata
            user_metadata = None
        self.user_metadata = dict(user_metadata or {}) if isinstance(user_metadata, dict) else {}
        self.project_id = project_id
        self.config = load_config(config)
        self._session = ModalSession(self.config) if self.config.modal.enabled else RaySession(self.config)

    @staticmethod
    def _looks_like_config(value: Any) -> bool:
        if isinstance(value, (str, RuntimeConfig)):
            return True
        if not isinstance(value, dict):
            return False
        config_keys = {
            "ray",
            "model",
            "lora",
            "model_configs",
            "resources",
            "modal",
            "checkpoint_root",
            "supported_models",
        }
        return any(key in value for key in config_keys)

    def get_server_capabilities(self) -> GetServerCapabilitiesResponse:
        supported = self.config.supported_model_names()
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

    def get_server_capabilities_async(self) -> GetServerCapabilitiesResponse:
        return self.get_server_capabilities()

    def create_lora_training_client(
        self,
        base_model: str | None = None,
        rank: int | None = None,
        seed: int | None = None,
        train_mlp: bool | None = None,
        train_attn: bool | None = None,
        train_unembed: bool | None = None,
        user_metadata: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TrainingClient:
        del kwargs
        run_metadata = self._merged_metadata(metadata, user_metadata)
        target_modules = None
        if train_mlp is not None or train_attn is not None or train_unembed is not None:
            target_modules = lora_target_modules_for_flags(
                train_mlp=True if train_mlp is None else train_mlp,
                train_attn=True if train_attn is None else train_attn,
                train_unembed=True if train_unembed is None else train_unembed,
            )
        session_id, actor = self._session.create_training_actor(
            base_model=base_model,
            lora_rank=rank,
            seed=seed,
            target_modules=target_modules,
            metadata=run_metadata,
        )
        return TrainingClient(session_id=session_id, actor=actor, service=self)

    def create_lora_training_client_async(self, *args, **kwargs) -> TrainingClient:
        return self.create_lora_training_client(*args, **kwargs)

    def create_sampling_client(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: Any | None = None,
        replicas: int | None = None,
        **kwargs,
    ) -> SamplingClient:
        del retry_config, kwargs
        session_id, actors = self._session.create_sampler_actors(
            base_model=base_model,
            model_path=model_path,
            replicas=replicas,
        )
        return SamplingClient(session_id=session_id, actors=actors)

    def create_sampling_client_async(self, *args, **kwargs) -> SamplingClient:
        return self.create_sampling_client(*args, **kwargs)

    def create_training_client_from_state(
        self,
        path: str,
        user_metadata: dict[str, str] | None = None,
        **kwargs,
    ) -> TrainingClient:
        base_model = kwargs.pop("base_model", None)
        rank = kwargs.pop("rank", None)
        metadata = kwargs.pop("metadata", None)
        del kwargs
        session_id, actor = self._session.create_training_actor(
            base_model=base_model,
            lora_rank=rank,
            model_path=path,
            with_optimizer=False,
            metadata=self._merged_metadata(metadata, user_metadata),
        )
        return TrainingClient(session_id=session_id, actor=actor, service=self)

    def create_training_client_from_state_async(self, *args, **kwargs) -> TrainingClient:
        return self.create_training_client_from_state(*args, **kwargs)

    def create_training_client_from_state_with_optimizer(
        self,
        path: str,
        user_metadata: dict[str, str] | None = None,
        **kwargs,
    ) -> TrainingClient:
        base_model = kwargs.pop("base_model", None)
        rank = kwargs.pop("rank", None)
        metadata = kwargs.pop("metadata", None)
        del kwargs
        session_id, actor = self._session.create_training_actor(
            base_model=base_model,
            lora_rank=rank,
            model_path=path,
            with_optimizer=True,
            metadata=self._merged_metadata(metadata, user_metadata),
        )
        return TrainingClient(session_id=session_id, actor=actor, service=self)

    def create_training_client_from_state_with_optimizer_async(self, *args, **kwargs) -> TrainingClient:
        return self.create_training_client_from_state_with_optimizer(*args, **kwargs)

    def create_rest_client(self):
        from ray_unsloth.clients.rest import RestClient

        return RestClient(config=self.config)

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()

    def _merged_metadata(
        self,
        metadata: dict[str, Any] | None = None,
        user_metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if self.project_id is not None:
            merged["project_id"] = self.project_id
        merged.update(self.user_metadata)
        if metadata:
            merged.update(metadata)
        if user_metadata:
            merged.update(user_metadata)
        return merged
