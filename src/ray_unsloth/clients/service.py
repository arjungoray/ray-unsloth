"""Main entry point for Ray Unsloth sessions."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from ray_unsloth.checkpoints import read_manifest, validate_restore_manifest
from ray_unsloth.clients._kwargs import warn_ignored
from ray_unsloth.clients.sampling import SamplingClient
from ray_unsloth.clients.training import TrainingClient
from ray_unsloth.config import RuntimeConfig, load_config, lora_target_modules_for_flags
from ray_unsloth.losses import loss_names
from ray_unsloth.plugins import load_entry_point_plugins
from ray_unsloth.providers import get_provider, resolve_provider_name
from ray_unsloth.runtime.modal import ModalSession
from ray_unsloth.runtime.ray import RaySession
from ray_unsloth.types import GetServerCapabilitiesResponse


class _TrainingRunContext:
    def __init__(self, service: ServiceClient, *, name: str | None, client_kwargs: dict[str, Any]):
        self._service = service
        self._name = name
        self._client_kwargs = dict(client_kwargs)
        self._client: TrainingClient | None = None

    def _apply_name(self, client: TrainingClient) -> None:
        if self._name is None:
            return
        store = getattr(self._service, "_store", None)
        run_id = getattr(client, "run_id", None)
        if store is None or run_id is None:
            return
        store.update_run(run_id, name=self._name)

    async def _create_client_async(self) -> TrainingClient:
        client = await self._service.create_lora_training_client_async(**self._client_kwargs)
        self._apply_name(client)
        return client

    def _create_client(self) -> TrainingClient:
        client = self._service.create_lora_training_client(**self._client_kwargs)
        self._apply_name(client)
        return client

    def __enter__(self) -> TrainingClient:
        self._client = self._create_client()
        return self._client

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc, tb
        if self._client is not None:
            self._client.close(status="failed" if exc_type is not None else "completed")
        return False

    async def __aenter__(self) -> TrainingClient:
        self._client = await self._create_client_async()
        return self._client

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        del exc, tb
        if self._client is not None:
            if exc_type is None:
                await asyncio.to_thread(self._client.close, status="completed")
            else:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self._client.close, status="failed")
        return False


class ServiceClient:
    """Main control-plane handle, matching Tinker's client-construction layer."""

    def __init__(
        self,
        user_metadata: dict[str, str] | str | RuntimeConfig | dict[str, Any] | None = None,
        project_id: str | None = None,
        config: str | RuntimeConfig | dict[str, Any] | None = None,
        base_url: str | None = None,
    ):
        if base_url is not None:
            warn_ignored(
                {"base_url": base_url},
                method="ServiceClient.__init__",
                accepted=("user_metadata", "project_id", "config"),
            )
        if config is None and self._looks_like_config(user_metadata):
            config = user_metadata
            user_metadata = None
        self.user_metadata = dict(user_metadata or {}) if isinstance(user_metadata, dict) else {}
        self.project_id = project_id
        self.config = load_config(config)
        load_entry_point_plugins()
        self.provider_name = resolve_provider_name(self.config)
        self._session = self._create_session(self.provider_name)
        self._store = None
        self._open_run_ids: list[str] = []
        if self.config.tracking:
            from ray_unsloth.store import RunStore

            self._store = RunStore(self.config.store_root)

    def _create_session(self, provider_name: str):
        # RaySession/ModalSession stay module attributes so tests (and callers)
        # can monkeypatch them; every other provider resolves via the registry.
        if provider_name == "local-ray":
            return RaySession(self.config)
        if provider_name == "modal":
            return ModalSession(self.config)
        return get_provider(provider_name).connect(self.config)

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
            "speed",
            "modal",
            "checkpoint_root",
            "supported_models",
            "provider",
            "provider_options",
            "plugins",
            "run_name",
            "tracking",
        }
        return any(key in value for key in config_keys)

    def get_server_capabilities(self) -> GetServerCapabilitiesResponse:
        supported = self.config.supported_model_names()
        return GetServerCapabilitiesResponse(
            supported_models=supported,
            supports_multi_trainer=True,
            max_sampler_replicas=self.config.resources.sampler_replicas,
            max_concurrent_trainers=self.config.resources.trainer_replicas,
            features={
                "losses": loss_names(),
                "checkpointing": True,
                "ray_namespace": self.config.ray.namespace,
                "runtime_backend": self.provider_name,
                "trainer_replicas": self.config.resources.trainer_replicas,
                "speed": {
                    "profile": self.config.speed.profile,
                    "padding_free": self.config.speed.padding_free,
                    "sample_packing": self.config.speed.sample_packing,
                    "optimizer": self.config.speed.optimizer,
                    "vllm_standby": self.config.speed.vllm_standby,
                    "flash_attention_2": self.config.speed.flash_attention_2,
                    "live_policy_sampling": self.config.speed.live_policy_sampling,
                },
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
        run_metadata = self._merged_metadata(metadata, user_metadata)
        target_modules = None
        if train_mlp is not None or train_attn is not None or train_unembed is not None:
            target_modules = lora_target_modules_for_flags(
                train_mlp=True if train_mlp is None else train_mlp,
                train_attn=True if train_attn is None else train_attn,
                train_unembed=True if train_unembed is None else train_unembed,
            )
        warn_ignored(
            kwargs,
            method="ServiceClient.create_lora_training_client",
            accepted=(
                "base_model",
                "rank",
                "seed",
                "train_mlp",
                "train_attn",
                "train_unembed",
                "user_metadata",
                "metadata",
            ),
        )
        session_id, actor = self._session.create_training_actor(
            base_model=base_model,
            lora_rank=rank,
            seed=seed,
            target_modules=target_modules,
            metadata=run_metadata,
        )
        recorder = self._start_run(
            session_id=session_id,
            base_model=base_model,
            rank=rank,
            metadata=run_metadata,
        )
        return TrainingClient(session_id=session_id, actor=actor, service=self, recorder=recorder)

    async def create_lora_training_client_async(self, *args, **kwargs) -> TrainingClient:
        return await asyncio.to_thread(self.create_lora_training_client, *args, **kwargs)

    def create_sampling_client(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: Any | None = None,
        replicas: int | None = None,
        **kwargs,
    ) -> SamplingClient:
        if retry_config is not None:
            warn_ignored(
                {"retry_config": retry_config},
                method="ServiceClient.create_sampling_client",
                accepted=("model_path", "base_model", "replicas"),
            )
        warn_ignored(
            kwargs,
            method="ServiceClient.create_sampling_client",
            accepted=("model_path", "base_model", "retry_config", "replicas"),
        )
        session_id, actors = self._session.create_sampler_actors(
            base_model=base_model,
            model_path=model_path,
            replicas=replicas,
        )
        return SamplingClient(session_id=session_id, actors=actors)

    async def create_sampling_client_async(self, *args, **kwargs) -> SamplingClient:
        return await asyncio.to_thread(self.create_sampling_client, *args, **kwargs)

    def create_training_client_from_state(
        self,
        path: str,
        user_metadata: dict[str, str] | None = None,
        **kwargs,
    ) -> TrainingClient:
        base_model = kwargs.pop("base_model", None)
        rank = kwargs.pop("rank", None)
        metadata = kwargs.pop("metadata", None)
        warn_ignored(
            kwargs,
            method="ServiceClient.create_training_client_from_state",
            accepted=("base_model", "rank", "metadata"),
        )
        self._validate_checkpoint_for_restore(path, base_model=base_model, rank=rank)
        run_metadata = self._merged_metadata(metadata, user_metadata)
        session_id, actor = self._session.create_training_actor(
            base_model=base_model,
            lora_rank=rank,
            model_path=path,
            with_optimizer=False,
            metadata=run_metadata,
        )
        recorder = self._start_run(
            session_id=session_id,
            base_model=base_model,
            rank=rank,
            metadata=run_metadata,
            restored_from=path,
        )
        return TrainingClient(session_id=session_id, actor=actor, service=self, recorder=recorder)

    async def create_training_client_from_state_async(self, *args, **kwargs) -> TrainingClient:
        return await asyncio.to_thread(self.create_training_client_from_state, *args, **kwargs)

    def create_training_client_from_state_with_optimizer(
        self,
        path: str,
        user_metadata: dict[str, str] | None = None,
        **kwargs,
    ) -> TrainingClient:
        base_model = kwargs.pop("base_model", None)
        rank = kwargs.pop("rank", None)
        metadata = kwargs.pop("metadata", None)
        warn_ignored(
            kwargs,
            method="ServiceClient.create_training_client_from_state_with_optimizer",
            accepted=("base_model", "rank", "metadata"),
        )
        self._validate_checkpoint_for_restore(path, base_model=base_model, rank=rank)
        run_metadata = self._merged_metadata(metadata, user_metadata)
        session_id, actor = self._session.create_training_actor(
            base_model=base_model,
            lora_rank=rank,
            model_path=path,
            with_optimizer=True,
            metadata=run_metadata,
        )
        recorder = self._start_run(
            session_id=session_id,
            base_model=base_model,
            rank=rank,
            metadata=run_metadata,
            restored_from=path,
        )
        return TrainingClient(session_id=session_id, actor=actor, service=self, recorder=recorder)

    async def create_training_client_from_state_with_optimizer_async(self, *args, **kwargs) -> TrainingClient:
        return await asyncio.to_thread(self.create_training_client_from_state_with_optimizer, *args, **kwargs)

    def training_run(self, name: str | None = None, **client_kwargs: Any):
        return _TrainingRunContext(self, name=name, client_kwargs=client_kwargs)

    def create_rest_client(self):
        from ray_unsloth.clients.rest import RestClient

        return RestClient(config=self.config)

    def attach_sampler_download_url(self, response):
        """Fill the download URL on a SamplerDownloadResponse if the session can serve one."""
        builder = getattr(self._session, "build_sampler_download_url", None)
        if callable(builder) and getattr(response, "url", None) is None:
            response.url = builder(response.archive_relpath, response.token, response.expires_at)
        return response

    def close(self) -> None:
        if self._store is not None:
            for run_id in self._open_run_ids:
                try:
                    record = self._store.get_run(run_id)
                    if record is not None and record.status == "running":
                        self._store.update_run(run_id, status="completed")
                except Exception:
                    pass
            self._open_run_ids.clear()
        close = getattr(self._session, "close", None)
        if callable(close):
            close()

    def _start_run(
        self,
        *,
        session_id: str,
        base_model: str | None,
        rank: int | None,
        metadata: dict[str, Any],
        restored_from: str | None = None,
    ):
        """Create a run record + recorder for a new training client (best-effort)."""
        if self._store is None:
            return None
        try:
            from dataclasses import asdict

            model_config, lora_config = self.config.resolve_model_configs(base_model)
            resolved_model = model_config.base_model
            resolved_rank = rank if rank is not None else lora_config.rank
            record = self._store.create_run(
                name=self.config.run_name,
                provider=self.provider_name,
                base_model=resolved_model,
                lora_rank=resolved_rank,
                session_id=session_id,
                metadata=metadata,
                config={
                    "model": asdict(model_config),
                    "lora": asdict(lora_config),
                    "provider": self.provider_name,
                    "checkpoint_root": self.config.checkpoint_root,
                },
            )
            self._open_run_ids.append(record.id)
            from ray_unsloth.recording import RunRecorder

            recorder = RunRecorder(self._store, record.id, base_model=resolved_model)
            if restored_from is not None:
                recorder.note_loaded_checkpoint(restored_from)
            return recorder
        except Exception:
            return None

    def _validate_checkpoint_for_restore(
        self,
        path: str,
        *,
        base_model: str | None,
        rank: int | None,
    ) -> None:
        """Validate a checkpoint manifest against the resolved config before restoring.

        Reads the checkpoint ``manifest.json`` and compares ``base_model``, ``lora.rank``,
        and ``lora.target_modules`` against the active configuration (honoring per-call
        ``base_model``/``rank`` overrides). Raises :class:`CheckpointError` on a missing or
        malformed manifest or any mismatch, before any actor is created or weights loaded.
        """
        manifest = read_manifest(path)
        model_config, lora_config = self.config.resolve_model_configs(base_model)
        expected_rank = rank if rank is not None else lora_config.rank
        validate_restore_manifest(
            manifest,
            path=path,
            base_model=model_config.base_model,
            lora_rank=expected_rank,
            target_modules=lora_config.target_modules,
        )

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
