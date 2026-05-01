"""Ray actor that owns one trainable Unsloth model."""

from __future__ import annotations

from typing import Any

from ray_unsloth.config import LoRAConfig, ModelConfig
from ray_unsloth.runtime.unsloth import UnslothEngine
from ray_unsloth.types import AdamParams, CustomLoss, Datum, ModelInput, SamplingParams


class TrainerActorImpl:
    def __init__(
        self,
        *,
        session_id: str,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        checkpoint_root: str,
        model_path: str | None = None,
        with_optimizer: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.engine = UnslothEngine(
            session_id=session_id,
            model_config=model_config,
            lora_config=lora_config,
            checkpoint_root=checkpoint_root,
            model_path=model_path,
            with_optimizer=with_optimizer,
            metadata=metadata or {},
        )

    def get_tokenizer(self):
        return self.engine.get_tokenizer()

    def get_info(self):
        return self.engine.get_info()

    def register_custom_loss(self, name: str, loss_fn: CustomLoss) -> None:
        self.engine.register_custom_loss(name, loss_fn)

    def compute_logprobs(self, prompt):
        return self.engine.compute_logprobs(prompt)

    def sample(
        self,
        prompt: ModelInput | list[int],
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ):
        return self.engine.sample(
            prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
        )

    def get_base_model(self) -> str:
        return self.engine.model_config.base_model

    def forward(self, data: list[Datum], loss_fn: str = "cross_entropy"):
        return self.engine.forward(data, loss_fn=loss_fn)

    def forward_backward(
        self,
        data: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, Any] | None = None,
    ):
        return self.engine.forward_backward(data, loss_fn=loss_fn, loss_fn_config=loss_fn_config)

    def forward_backward_custom(
        self,
        data: list[Datum],
        loss_fn,
        loss_fn_config: dict[str, Any] | None = None,
    ):
        return self.engine.forward_backward_custom(data, loss_fn=loss_fn, loss_fn_config=loss_fn_config)

    def optim_step(self, adam_params: AdamParams):
        return self.engine.optim_step(adam_params)

    def save_state(self, path: str | None = None):
        return self.engine.save_state(path=path, include_optimizer=False)

    def save_state_with_optimizer(self, path: str | None = None):
        return self.engine.save_state(path=path, include_optimizer=True)

    def load_state(self, path: str):
        return self.engine.load_state(path, with_optimizer=False)

    def load_state_with_optimizer(self, path: str):
        return self.engine.load_state_with_optimizer(path)

    def save_weights_for_sampler(self, path: str | None = None):
        return self.engine.save_weights_for_sampler(path=path)


try:
    import ray

    TrainerActor = ray.remote(TrainerActorImpl)
except ImportError:  # pragma: no cover - import-time fallback
    TrainerActor = TrainerActorImpl
