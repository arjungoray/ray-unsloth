"""Tinker-shaped public request and response types.

The types intentionally stay lightweight and pickle-friendly because Ray will
move them between the local training loop and GPU actors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Generic, Iterable, Mapping, TypeVar

T = TypeVar("T")


class RayObjectFuture(Generic[T]):
    """Small future wrapper around a Ray ObjectRef.

    Tinker's client methods return futures. This wrapper gives callers a
    familiar `.result()` / `.get()` shape without exposing Ray as the public API.
    """

    def __init__(self, object_ref: Any):
        self.object_ref = object_ref

    def result(self, timeout: float | None = None) -> T:
        try:
            import ray
        except ImportError as exc:  # pragma: no cover - exercised without deps
            raise RuntimeError("Ray is required to resolve this future.") from exc
        if timeout is None:
            return ray.get(self.object_ref)
        return ray.get(self.object_ref, timeout=timeout)

    def get(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)


class ImmediateFuture(Generic[T]):
    """Future wrapper for synchronous local values."""

    def __init__(self, value: T):
        self._value = value

    def result(self, timeout: float | None = None) -> T:
        del timeout
        return self._value

    def get(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)


def future_from(value: Any) -> RayObjectFuture[Any] | ImmediateFuture[Any]:
    """Wrap Ray object refs and synchronous values behind one future protocol."""

    if hasattr(value, "hex") and value.__class__.__name__ == "ObjectRef":
        return RayObjectFuture(value)
    return ImmediateFuture(value)


@dataclass(slots=True)
class ModelInput:
    """Tokenized model input compatible with Tinker-style examples."""

    tokens: list[int]

    @classmethod
    def from_ints(cls, tokens: Iterable[int]) -> "ModelInput":
        return cls(tokens=list(tokens))

    def to_ints(self) -> list[int]:
        return list(self.tokens)


@dataclass(slots=True)
class Datum:
    """One training/evaluation datum.

    `loss_fn_inputs` holds task-specific fields such as `labels`, `label_mask`,
    or `target_tokens`. Keeping it structured avoids baking an SFT dataset shape
    into the client API.
    """

    model_input: ModelInput
    loss_fn_inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


@dataclass(slots=True)
class AdamParams:
    learning_rate: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float | None = None


@dataclass(slots=True)
class GeneratedSequence:
    tokens: list[int]
    text: str | None = None
    logprobs: list[float | None] | None = None
    finish_reason: str | None = None


@dataclass(slots=True)
class SampleResponse:
    sequences: list[GeneratedSequence]


@dataclass(slots=True)
class ForwardOutput:
    loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    logprobs: list[list[float | None]] | None = None


@dataclass(slots=True)
class ForwardBackwardOutput:
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class OptimStepResult:
    step: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CheckpointRef:
    path: str
    step: int | None = None
    has_optimizer: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SaveWeightsForSamplerResponse:
    path: str
    checkpoint: CheckpointRef


@dataclass(slots=True)
class TrainingClientInfo:
    session_id: str
    base_model: str
    lora_rank: int
    step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GetServerCapabilitiesResponse:
    supported_models: list[str]
    supports_lora: bool = True
    supports_custom_loss: bool = True
    supports_multi_sampler: bool = True
    supports_multi_trainer: bool = False
    max_sampler_replicas: int = 1
    features: dict[str, Any] = field(default_factory=dict)


CustomLoss = Callable[[Any, list[Datum], Mapping[str, Any]], tuple[Any, dict[str, float]]]


def to_plain_data(value: Any) -> Any:
    """Convert nested dataclasses into JSON/YAML-friendly primitives."""

    if is_dataclass(value):
        return {key: to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_data(item) for item in value]
    return value
