"""Ray actor that owns one inference-oriented Unsloth model."""

from __future__ import annotations

from ray_unsloth.config import LoRAConfig, ModelConfig
from ray_unsloth.runtime.unsloth import UnslothEngine
from ray_unsloth.types import ModelInput, SamplingParams


class SamplerActorImpl:
    def __init__(
        self,
        *,
        session_id: str,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        checkpoint_root: str,
        model_path: str | None = None,
    ) -> None:
        self.engine = UnslothEngine(
            session_id=session_id,
            model_config=model_config,
            lora_config=lora_config,
            checkpoint_root=checkpoint_root,
            model_path=model_path,
            with_optimizer=False,
        )

    def sample(
        self,
        prompt: ModelInput | list[int],
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
        **kwargs,
    ):
        del kwargs
        return self.engine.sample(prompt, num_samples=num_samples, sampling_params=sampling_params)

    def compute_logprobs(self, prompt: ModelInput | list[int]):
        return self.engine.compute_logprobs(prompt)

    def get_tokenizer(self):
        return self.engine.get_tokenizer()

    def get_base_model(self) -> str:
        return self.engine.model_config.base_model


try:
    import ray

    SamplerActor = ray.remote(SamplerActorImpl)
except ImportError:  # pragma: no cover - import-time fallback
    SamplerActor = SamplerActorImpl
