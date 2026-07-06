"""Training client facade."""

from __future__ import annotations

from typing import Any

from ray_unsloth.clients._kwargs import warn_ignored
from ray_unsloth.clients._remote import call, call_async, resolve
from ray_unsloth.clients.sampling import SamplingClient
from ray_unsloth.types import AdamParams, CustomLoss, Datum, FutureValueProxy, ModelInput


class TrainingClient:
    """Tinker-shaped client for trainable model sessions."""

    def __init__(self, *, session_id: str, actor: Any, service: Any, recorder: Any | None = None):
        self.session_id = session_id
        self._actor = actor
        self._service = service
        self._recorder = recorder

    @property
    def run_id(self) -> str | None:
        """The run-store id this client records into, when tracking is enabled."""
        return self._recorder.run_id if self._recorder is not None else None

    def _record_fb(self, future: Any, loss_fn: str) -> Any:
        if self._recorder is None:
            return future
        return self._recorder.wrap_forward_backward(future, loss_fn=loss_fn)

    def _record_optim(self, future: Any) -> Any:
        if self._recorder is None:
            return future
        return self._recorder.wrap_optim_step(future)

    def _record_save(self, future: Any, *, kind: str, has_optimizer: bool = False) -> Any:
        if self._recorder is None:
            return future
        return self._recorder.wrap_save(future, kind=kind, has_optimizer=has_optimizer)

    def __await__(self):
        async def _self():
            return self

        return _self().__await__()

    def get_tokenizer(self):
        return FutureValueProxy(resolve(call(self._actor, "get_tokenizer")))

    def get_info(self):
        return FutureValueProxy(resolve(call(self._actor, "get_info")))

    async def get_info_async(self):
        return await call_async(self._actor, "get_info").result_async()

    def compute_logprobs(self, prompt: ModelInput | list[int]):
        return call(self._actor, "compute_logprobs", prompt)

    def forward(self, data: list[Datum], loss_fn: str = "cross_entropy"):
        return call(self._actor, "forward", data, loss_fn)

    def forward_async(self, data: list[Datum], loss_fn: str = "cross_entropy"):
        return call_async(self._actor, "forward", data, loss_fn)

    def forward_backward(
        self,
        data: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, Any] | None = None,
    ):
        return self._record_fb(
            call(self._actor, "forward_backward", data, loss_fn, loss_fn_config),
            loss_fn,
        )

    def forward_backward_async(
        self,
        data: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, Any] | None = None,
    ):
        return self._record_fb(
            call_async(self._actor, "forward_backward", data, loss_fn, loss_fn_config),
            loss_fn,
        )

    def register_custom_loss(self, name: str, loss_fn: CustomLoss):
        return call(self._actor, "register_custom_loss", name, loss_fn)

    def forward_backward_custom(
        self,
        data: list[Datum],
        loss_fn,
        loss_fn_config: dict[str, Any] | None = None,
        *,
        loss_type_input: str = "logprobs",
    ):
        if loss_type_input != "logprobs":
            warn_ignored(
                {"loss_type_input": loss_type_input},
                method="TrainingClient.forward_backward_custom",
                accepted=("data", "loss_fn", "loss_fn_config"),
            )
        return call(self._actor, "forward_backward_custom", data, loss_fn, loss_fn_config)

    def forward_backward_custom_async(
        self,
        data: list[Datum],
        loss_fn,
        loss_fn_config: dict[str, Any] | None = None,
        *,
        loss_type_input: str = "logprobs",
    ):
        if loss_type_input != "logprobs":
            warn_ignored(
                {"loss_type_input": loss_type_input},
                method="TrainingClient.forward_backward_custom_async",
                accepted=("data", "loss_fn", "loss_fn_config"),
            )
        return call_async(self._actor, "forward_backward_custom", data, loss_fn, loss_fn_config)

    def optim_step(self, adam_params: AdamParams):
        return self._record_optim(call(self._actor, "optim_step", adam_params))

    def optim_step_async(self, adam_params: AdamParams):
        return self._record_optim(call_async(self._actor, "optim_step", adam_params))

    def save_state(self, path: str | None = None, ttl_seconds: int | None = None, *, name: str | None = None):
        if ttl_seconds is not None:
            warn_ignored(
                {"ttl_seconds": ttl_seconds},
                method="TrainingClient.save_state",
                accepted=("path", "name"),
            )
        path = path or name
        return self._record_save(call(self._actor, "save_state", path), kind="training_state")

    def save_state_async(self, path: str | None = None, ttl_seconds: int | None = None, *, name: str | None = None):
        if ttl_seconds is not None:
            warn_ignored(
                {"ttl_seconds": ttl_seconds},
                method="TrainingClient.save_state_async",
                accepted=("path", "name"),
            )
        path = path or name
        return self._record_save(call_async(self._actor, "save_state", path), kind="training_state")

    def save_state_with_optimizer(
        self,
        path: str | None = None,
        ttl_seconds: int | None = None,
        *,
        name: str | None = None,
    ):
        if ttl_seconds is not None:
            warn_ignored(
                {"ttl_seconds": ttl_seconds},
                method="TrainingClient.save_state_with_optimizer",
                accepted=("path", "name"),
            )
        path = path or name
        return self._record_save(
            call(self._actor, "save_state_with_optimizer", path),
            kind="training_state",
            has_optimizer=True,
        )

    def save_state_with_optimizer_async(
        self,
        path: str | None = None,
        ttl_seconds: int | None = None,
        *,
        name: str | None = None,
    ):
        if ttl_seconds is not None:
            warn_ignored(
                {"ttl_seconds": ttl_seconds},
                method="TrainingClient.save_state_with_optimizer_async",
                accepted=("path", "name"),
            )
        path = path or name
        return self._record_save(
            call_async(self._actor, "save_state_with_optimizer", path),
            kind="training_state",
            has_optimizer=True,
        )

    def load_state(self, path: str):
        future = call(self._actor, "load_state", path)
        if self._recorder is not None:
            self._recorder.note_loaded_checkpoint(path)
        return future

    def load_state_async(self, path: str):
        future = call_async(self._actor, "load_state", path)
        if self._recorder is not None:
            self._recorder.note_loaded_checkpoint(path)
        return future

    def load_state_with_optimizer(self, path: str):
        future = call(self._actor, "load_state_with_optimizer", path)
        if self._recorder is not None:
            self._recorder.note_loaded_checkpoint(path)
        return future

    def load_state_with_optimizer_async(self, path: str):
        future = call_async(self._actor, "load_state_with_optimizer", path)
        if self._recorder is not None:
            self._recorder.note_loaded_checkpoint(path)
        return future

    def save_weights_for_sampler(
        self,
        path: str | None = None,
        ttl_seconds: int | None = None,
        *,
        name: str | None = None,
    ):
        if ttl_seconds is not None:
            warn_ignored(
                {"ttl_seconds": ttl_seconds},
                method="TrainingClient.save_weights_for_sampler",
                accepted=("path", "name"),
            )
        path = path or name
        return self._record_save(call(self._actor, "save_weights_for_sampler", path), kind="sampler")

    def save_weights_for_sampler_async(
        self,
        path: str | None = None,
        ttl_seconds: int | None = None,
        *,
        name: str | None = None,
    ):
        if ttl_seconds is not None:
            warn_ignored(
                {"ttl_seconds": ttl_seconds},
                method="TrainingClient.save_weights_for_sampler_async",
                accepted=("path", "name"),
            )
        path = path or name
        return self._record_save(call_async(self._actor, "save_weights_for_sampler", path), kind="sampler")

    def save_sampler_with_download_url(
        self,
        path: str | None = None,
        *,
        name: str | None = None,
        ttl_seconds: int = 3600,
    ):
        path = path or name
        response = resolve(
            self._record_save(
                call(self._actor, "save_sampler_with_download_url", path, ttl_seconds),
                kind="sampler",
            )
        )
        if response is not None and self._service is not None:
            attach = getattr(self._service, "attach_sampler_download_url", None)
            if callable(attach):
                response = attach(response)
        return response

    async def save_sampler_with_download_url_async(
        self,
        path: str | None = None,
        *,
        name: str | None = None,
        ttl_seconds: int = 3600,
    ):
        path = path or name
        future = self._record_save(
            call_async(self._actor, "save_sampler_with_download_url", path, ttl_seconds),
            kind="sampler",
        )
        response = await future.result_async()
        if response is not None and self._service is not None:
            attach = getattr(self._service, "attach_sampler_download_url", None)
            if callable(attach):
                response = attach(response)
        return response

    def create_sampling_client(self, model_path: str, retry_config: Any | None = None) -> SamplingClient:
        return self._service.create_sampling_client(model_path=model_path, retry_config=retry_config)

    def create_sampling_client_async(self, model_path: str, retry_config: Any | None = None) -> SamplingClient:
        return self.create_sampling_client(model_path=model_path, retry_config=retry_config)

    def create_live_sampling_client(self, name: str = "live-policy") -> SamplingClient:
        return SamplingClient(session_id=f"{self.session_id}-{name}", actors=[self._actor])

    def create_live_sampling_client_async(self, name: str = "live-policy") -> SamplingClient:
        return self.create_live_sampling_client(name=name)

    def save_weights_and_get_sampling_client(
        self,
        path: str | None = None,
        *,
        name: str | None = None,
        retry_config: Any | None = None,
        replicas: int | None = None,
    ) -> SamplingClient:
        if replicas in (None, 1):
            if retry_config is not None:
                warn_ignored(
                    {"retry_config": retry_config},
                    method="TrainingClient.save_weights_and_get_sampling_client",
                    accepted=("path", "name", "replicas"),
                )
            if path is not None or name is not None:
                resolve(self.save_weights_for_sampler(path, name=name))
            return SamplingClient(session_id=f"{self.session_id}-sampler", actors=[self._actor])
        saved = resolve(self.save_weights_for_sampler(path, name=name))
        return self._service.create_sampling_client(
            model_path=saved.path,
            retry_config=retry_config,
            replicas=replicas,
        )

    async def save_weights_and_get_sampling_client_async(
        self,
        path: str | None = None,
        *,
        name: str | None = None,
        retry_config: Any | None = None,
        replicas: int | None = None,
    ) -> SamplingClient:
        if replicas in (None, 1):
            if retry_config is not None:
                warn_ignored(
                    {"retry_config": retry_config},
                    method="TrainingClient.save_weights_and_get_sampling_client_async",
                    accepted=("path", "name", "replicas"),
                )
            if path is not None or name is not None:
                await self.save_weights_for_sampler_async(path, name=name).result_async()
            return SamplingClient(session_id=f"{self.session_id}-sampler", actors=[self._actor])
        saved = await self.save_weights_for_sampler_async(path, name=name).result_async()
        return self._service.create_sampling_client(
            model_path=saved.path,
            retry_config=retry_config,
            replicas=replicas,
        )
