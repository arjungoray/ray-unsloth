"""Torch-free reward scoring helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

RewardFn = Callable[..., float]


@dataclass(slots=True)
class RewardBreakdown:
    """A scored sample with the total reward and per-term contributions."""

    total: float
    terms: dict[str, float]


@dataclass(slots=True)
class RubricTerm:
    """One scoring term in a rubric."""

    name: str
    fn: RewardFn
    weight: float
    z_normalize: bool = True
    override_below: float | None = None


@dataclass(slots=True)
class Rubric:
    """Score completions using a small set of reusable reward terms."""

    terms: list[RubricTerm]
    _buffers: dict[str, deque[float]] = field(default_factory=dict, init=False, repr=False)

    def score(self, samples: list[dict[str, Any]]) -> list[RewardBreakdown]:
        """Score samples and return weighted term breakdowns."""

        breakdowns: list[RewardBreakdown] = []
        for sample in samples:
            prompt = sample.get("prompt")
            completion_text = str(sample.get("completion_text", ""))
            completion_tokens = [int(token) for token in sample.get("completion_tokens", [])]
            context = self._sample_context(sample)

            total = 0.0
            terms: dict[str, float] = {}
            override_total: float | None = None
            for term in self.terms:
                raw = float(
                    term.fn(
                        prompt=prompt,
                        completion_text=completion_text,
                        completion_tokens=completion_tokens,
                        **context,
                    )
                )
                buffer = self._buffers.setdefault(term.name, deque(maxlen=512))
                buffer.append(raw)
                if term.override_below is not None and raw <= term.override_below:
                    override_total = -1.0
                if term.z_normalize:
                    mean = sum(buffer) / len(buffer)
                    variance = sum((value - mean) ** 2 for value in buffer) / len(buffer)
                    contribution = ((raw - mean) / (sqrt(variance) + 1e-6)) * float(term.weight)
                else:
                    contribution = raw * float(term.weight)
                terms[term.name] = contribution
                total += contribution

            total = override_total if override_total is not None else max(-2.0, min(2.0, total))
            breakdowns.append(RewardBreakdown(total=total, terms=terms))
        return breakdowns

    @staticmethod
    def _sample_context(sample: dict[str, Any]) -> dict[str, Any]:
        context: dict[str, Any] = {}
        if isinstance(sample.get("context"), dict):
            context.update(sample["context"])
        for key, value in sample.items():
            if key not in {"prompt", "completion_text", "completion_tokens", "context"}:
                context.setdefault(key, value)
        return context


__all__ = [
    "RewardBreakdown",
    "RewardFn",
    "Rubric",
    "RubricTerm",
]
