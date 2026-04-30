"""Training client facade."""

from __future__ import annotations

from typing import Any

from ray_unsloth.clients._remote import call, resolve
from ray_unsloth.clients.sampling import SamplingClient
from ray_unsloth.types import AdamParams, CustomLoss, Datum, ModelInput


class TrainingClient:
    """Tinker-shaped client for trainable model sessions."""

    def __init__(self, *, session_id: str, actor: Any, service: Any):
        self.session_id = session_id
        self._actor = actor
        self._service = service

    def get_tokenizer(self):
        return call(self._actor, "get_tokenizer")

    def get_info(self):
        return call(self._actor, "get_info")

    def compute_logprobs(self, prompt: ModelInput | list[int]):
        return call(self._actor, "compute_logprobs", prompt)

    def forward(self, data: list[Datum], loss_fn: str = "cross_entropy"):
        return call(self._actor, "forward", data, loss_fn)

    def forward_async(self, data: list[Datum], loss_fn: str = "cross_entropy"):
        return self.forward(data, loss_fn=loss_fn)

    def forward_backward(
        self,
        data: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, Any] | None = None,
    ):
        return call(self._actor, "forward_backward", data, loss_fn, loss_fn_config)

    def forward_backward_async(
        self,
        data: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, Any] | None = None,
    ):
        return self.forward_backward(data, loss_fn=loss_fn, loss_fn_config=loss_fn_config)

    def register_custom_loss(self, name: str, loss_fn: CustomLoss):
        return call(self._actor, "register_custom_loss", name, loss_fn)

    def forward_backward_custom(
        self,
        data: list[Datum],
        loss_fn,
        loss_fn_config: dict[str, Any] | None = None,
    ):
        return call(self._actor, "forward_backward_custom", data, loss_fn, loss_fn_config)

    def forward_backward_custom_async(
        self,
        data: list[Datum],
        loss_fn,
        loss_fn_config: dict[str, Any] | None = None,
    ):
        return self.forward_backward_custom(data, loss_fn=loss_fn, loss_fn_config=loss_fn_config)

    def optim_step(self, adam_params: AdamParams):
        return call(self._actor, "optim_step", adam_params)

    def optim_step_async(self, adam_params: AdamParams):
        return self.optim_step(adam_params)

    def save_state(self, path: str | None = None, ttl_seconds: int | None = None):
        del ttl_seconds
        return call(self._actor, "save_state", path)

    def save_state_async(self, path: str | None = None, ttl_seconds: int | None = None):
        return self.save_state(path, ttl_seconds=ttl_seconds)

    def save_state_with_optimizer(self, path: str | None = None, ttl_seconds: int | None = None):
        del ttl_seconds
        return call(self._actor, "save_state_with_optimizer", path)

    def save_state_with_optimizer_async(self, path: str | None = None, ttl_seconds: int | None = None):
        return self.save_state_with_optimizer(path, ttl_seconds=ttl_seconds)

    def load_state(self, path: str):
        return call(self._actor, "load_state", path)

    def load_state_async(self, path: str):
        return self.load_state(path)

    def load_state_with_optimizer(self, path: str):
        return call(self._actor, "load_state_with_optimizer", path)

    def load_state_with_optimizer_async(self, path: str):
        return self.load_state_with_optimizer(path)

    def save_weights_for_sampler(self, path: str | None = None, ttl_seconds: int | None = None):
        del ttl_seconds
        return call(self._actor, "save_weights_for_sampler", path)

    def save_weights_for_sampler_async(self, path: str | None = None, ttl_seconds: int | None = None):
        return self.save_weights_for_sampler(path, ttl_seconds=ttl_seconds)

    def create_sampling_client(self, model_path: str, retry_config: Any | None = None) -> SamplingClient:
        return self._service.create_sampling_client(model_path=model_path, retry_config=retry_config)

    def create_sampling_client_async(self, model_path: str, retry_config: Any | None = None) -> SamplingClient:
        return self.create_sampling_client(model_path=model_path, retry_config=retry_config)

    def save_weights_and_get_sampling_client(
        self,
        path: str | None = None,
        *,
        retry_config: Any | None = None,
        replicas: int | None = None,
    ) -> SamplingClient:
        saved = resolve(self.save_weights_for_sampler(path))
        return self._service.create_sampling_client(
            model_path=saved.path,
            retry_config=retry_config,
            replicas=replicas,
        )

    def save_weights_and_get_sampling_client_async(
        self,
        path: str | None = None,
        *,
        retry_config: Any | None = None,
        replicas: int | None = None,
    ) -> SamplingClient:
        return self.save_weights_and_get_sampling_client(
            path,
            retry_config=retry_config,
            replicas=replicas,
        )
