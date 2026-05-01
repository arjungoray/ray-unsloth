"""Sampling client facade."""

from __future__ import annotations

from itertools import count
from typing import Any

from ray_unsloth.clients._remote import call, resolve
from ray_unsloth.types import FutureValueProxy, ModelInput, SamplingParams


class SamplingClient:
    """Tinker-shaped client for inference/rollout primitives."""

    def __init__(self, *, session_id: str, actors: list[Any]):
        if not actors:
            raise ValueError("SamplingClient requires at least one sampler actor")
        self.session_id = session_id
        self._actors = actors
        self._counter = count()

    def __await__(self):
        async def _self():
            return self

        return _self().__await__()

    def _next_actor(self):
        return self._actors[next(self._counter) % len(self._actors)]

    def sample(
        self,
        prompt: ModelInput | list[int],
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ):
        return call(
            self._next_actor(),
            "sample",
            prompt,
            num_samples,
            sampling_params,
            include_prompt_logprobs,
            topk_prompt_logprobs,
        )

    def compute_logprobs(self, prompt: ModelInput | list[int]):
        return call(self._next_actor(), "compute_logprobs", prompt)

    async def sample_async(
        self,
        prompt: ModelInput | list[int],
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ):
        return await call(
            self._next_actor(),
            "sample",
            prompt,
            num_samples,
            sampling_params,
            include_prompt_logprobs,
            topk_prompt_logprobs,
        ).result_async()

    async def compute_logprobs_async(self, prompt: ModelInput | list[int]):
        return await call(self._next_actor(), "compute_logprobs", prompt).result_async()

    def get_tokenizer(self):
        return FutureValueProxy(resolve(call(self._actors[0], "get_tokenizer")))

    def get_base_model(self):
        return FutureValueProxy(resolve(call(self._actors[0], "get_base_model")))

    async def get_base_model_async(self):
        return self.get_base_model().result()
