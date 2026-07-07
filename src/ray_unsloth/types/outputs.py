from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from ray_unsloth.types.inputs import Datum, ModelInput


@dataclass(slots=True, init=False)
class GeneratedSequence:
    stop_reason: str | None
    tokens_np: Any | None
    logprobs_np: Any | None
    _tokens_list: list[int] | None
    _logprobs_list: list[float | None] | None
    text: str | None
    finish_reason: str | None

    def __init__(
        self,
        tokens: Sequence[int] | None = None,
        text: str | None = None,
        logprobs: Sequence[float | None] | None = None,
        finish_reason: str | None = None,
        stop_reason: str | None = None,
        *,
        tokens_np: Any | None = None,
        logprobs_np: Any | None = None,
        _tokens_list: Sequence[int] | None = None,
        _logprobs_list: Sequence[float | None] | None = None,
    ) -> None:
        self.tokens_np = tokens_np
        self.logprobs_np = logprobs_np
        self._tokens_list = (
            list(tokens) if tokens is not None else (list(_tokens_list) if _tokens_list is not None else None)
        )
        self._logprobs_list = (
            list(logprobs) if logprobs is not None else (list(_logprobs_list) if _logprobs_list is not None else None)
        )
        self.text = text
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason
        if self.finish_reason is None and self.stop_reason is not None:
            self.finish_reason = self.stop_reason
        if self.stop_reason is None and self.finish_reason is not None:
            self.stop_reason = self.finish_reason

    @property
    def tokens(self) -> list[int]:
        if self._tokens_list is not None:
            return self._tokens_list
        if self.tokens_np is not None:
            return [int(token) for token in self.tokens_np.tolist()]
        return []

    @property
    def logprobs(self) -> list[float | None] | None:
        if self._logprobs_list is not None:
            return self._logprobs_list
        if self.logprobs_np is not None:
            return [float(value) for value in self.logprobs_np.tolist()]
        return None


SampledSequence = GeneratedSequence


@dataclass(slots=True)
class SampleResponse:
    sequences: Sequence[GeneratedSequence]
    type: str = "sample"
    prompt_logprobs: list[float | None] | None = None
    topk_prompt_logprobs: list[list[tuple[int, float]] | None] | None = None
    prompt_logprobs_np: Any | None = None
    topk_prompt_logprobs_np: Any | None = None

    def result(self, timeout: float | None = None) -> SampleResponse:
        del timeout
        return self

    async def result_async(self, timeout: float | None = None) -> SampleResponse:
        return self.result(timeout=timeout)

    def get(self, timeout: float | None = None) -> SampleResponse:
        return self.result(timeout=timeout)


@dataclass(slots=True)
class ForwardOutput:
    loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    logprobs: list[list[float | None]] | None = None
    loss_fn_outputs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ForwardBackwardOutput:
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    loss_fn_outputs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class OptimStepResult:
    step: int
    metrics: dict[str, float] = field(default_factory=dict)


OptimStepResponse = OptimStepResult


CustomLoss = Callable[[Any, list[Datum], Mapping[str, Any]], tuple[Any, dict[str, float]]]


def to_plain_data(value: Any) -> Any:
    """Convert nested dataclasses into JSON/YAML-friendly primitives."""

    if isinstance(value, ModelInput):
        return {"tokens": value.to_ints(), "chunks": to_plain_data(value.chunks)}
    if is_dataclass(value):
        return {item.name: to_plain_data(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_data(item) for item in value]
    return value


__all__ = [
    "CustomLoss",
    "ForwardBackwardOutput",
    "ForwardOutput",
    "GeneratedSequence",
    "OptimStepResponse",
    "OptimStepResult",
    "SampleResponse",
    "SampledSequence",
    "to_plain_data",
]
