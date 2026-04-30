"""Sampling client facade."""

from __future__ import annotations

from itertools import count
from typing import Any

from ray_unsloth.clients._remote import call
from ray_unsloth.types import ModelInput, SamplingParams


class SamplingClient:
    """Tinker-shaped client for inference/rollout primitives."""

    def __init__(self, *, session_id: str, actors: list[Any]):
        if not actors:
            raise ValueError("SamplingClient requires at least one sampler actor")
        self.session_id = session_id
        self._actors = actors
        self._counter = count()

    def _next_actor(self):
        return self._actors[next(self._counter) % len(self._actors)]

    def sample(
        self,
        prompt: ModelInput | list[int],
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
        **kwargs,
    ):
        return call(
            self._next_actor(),
            "sample",
            prompt,
            num_samples,
            sampling_params,
            **kwargs,
        )

    def compute_logprobs(self, prompt: ModelInput | list[int]):
        return call(self._next_actor(), "compute_logprobs", prompt)

    def get_tokenizer(self):
        return call(self._actors[0], "get_tokenizer")

    def get_base_model(self):
        return call(self._actors[0], "get_base_model")
