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

    def __await__(self):
        async def _resolve():
            return self.result()

        return _resolve().__await__()


class ImmediateFuture(Generic[T]):
    """Future wrapper for synchronous local values."""

    def __init__(self, value: T):
        self._value = value

    def result(self, timeout: float | None = None) -> T:
        del timeout
        return self._value

    def get(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def __await__(self):
        async def _resolve():
            return self.result()

        return _resolve().__await__()


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

    @classmethod
    def empty(cls) -> "ModelInput":
        return cls(tokens=[])

    def to_ints(self) -> list[int]:
        return list(self.tokens)

    @property
    def length(self) -> int:
        return len(self.tokens)

    def append(self, chunk: "ModelInput | Iterable[int] | int") -> "ModelInput":
        if isinstance(chunk, ModelInput):
            return ModelInput(self.tokens + chunk.to_ints())
        if isinstance(chunk, int):
            return self.append_int(chunk)
        return ModelInput(self.tokens + list(chunk))

    def append_int(self, token: int) -> "ModelInput":
        return ModelInput(self.tokens + [token])


@dataclass(slots=True)
class TensorData:
    """Small serializable tensor container compatible with Tinker-style Datum inputs."""

    data: Any
    dtype: str
    shape: list[int]


def _convert_tensor_data(value: Any) -> Any:
    if isinstance(value, TensorData):
        return value
    if isinstance(value, ModelInput):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        tensor = detach().cpu()
        return TensorData(data=tensor.tolist(), dtype=str(tensor.dtype).removeprefix("torch."), shape=list(tensor.shape))
    if hasattr(value, "tolist") and hasattr(value, "shape") and hasattr(value, "dtype"):
        return TensorData(data=value.tolist(), dtype=str(value.dtype), shape=list(value.shape))
    if isinstance(value, dict):
        return {key: _convert_tensor_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_convert_tensor_data(item) for item in value]
    if isinstance(value, tuple):
        return [_convert_tensor_data(item) for item in value]
    return value


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

    def __post_init__(self) -> None:
        self.loss_fn_inputs = _convert_tensor_data(self.loss_fn_inputs)


@dataclass(slots=True)
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


@dataclass(slots=True, init=False)
class AdamParams:
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    max_grad_norm: float | None

    def __init__(
        self,
        learning_rate: float,
        betas: tuple[float, float] | None = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float | None = None,
        *,
        beta1: float | None = None,
        beta2: float | None = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        if betas is None:
            betas = (0.9 if beta1 is None else beta1, 0.999 if beta2 is None else beta2)
        if grad_clip_norm is not None:
            max_grad_norm = None if grad_clip_norm == 0.0 else grad_clip_norm
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

    @property
    def beta1(self) -> float:
        return self.betas[0]

    @property
    def beta2(self) -> float:
        return self.betas[1]

    @property
    def grad_clip_norm(self) -> float:
        return 0.0 if self.max_grad_norm is None else self.max_grad_norm


@dataclass(slots=True)
class GeneratedSequence:
    tokens: list[int]
    text: str | None = None
    logprobs: list[float | None] | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if self.finish_reason is None and self.stop_reason is not None:
            self.finish_reason = self.stop_reason
        if self.stop_reason is None and self.finish_reason is not None:
            self.stop_reason = self.finish_reason


SampledSequence = GeneratedSequence


@dataclass(slots=True)
class SampleResponse:
    sequences: list[GeneratedSequence]
    type: str = "sample"
    prompt_logprobs: list[float | None] | None = None
    topk_prompt_logprobs: list[list[tuple[int, float]]] | None = None


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
class Checkpoint:
    path: str
    step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "checkpoint"


@dataclass(slots=True)
class TrainingRun:
    id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    type: str = "training_run"


@dataclass(slots=True)
class CheckpointsListResponse:
    checkpoints: list[Checkpoint]
    type: str = "checkpoints_list"


@dataclass(slots=True)
class TrainingRunsResponse:
    training_runs: list[TrainingRun]
    type: str = "training_runs"


@dataclass(slots=True)
class WeightsInfoResponse:
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "weights_info"


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
