"""Tinker-shaped public request and response types.

The types intentionally stay lightweight and pickle-friendly because Ray moves
them between the local training loop and GPU actors. The public surface mirrors
the Tinker SDK closely enough for cookbook examples to construct data without
local adapter glue.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from math import prod
from typing import Any, Awaitable, Callable, Generic, Iterable, Literal, Mapping, Sequence, TypeVar

T = TypeVar("T")


def _flatten_nested(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        flattened: list[Any] = []
        for item in value:
            if isinstance(item, (list, tuple)) or hasattr(item, "tolist"):
                flattened.extend(_flatten_nested(item))
            else:
                flattened.append(item)
        return flattened
    return [value]


def _reshape_flat(data: Sequence[Any], shape: Sequence[int] | None) -> Any:
    values = list(data)
    if shape is None or len(shape) <= 1:
        return values
    if prod(shape) != len(values):
        return values

    def _reshape(offset: int, dims: Sequence[int]) -> tuple[Any, int]:
        if len(dims) == 1:
            end = offset + dims[0]
            return values[offset:end], end
        rows = []
        for _ in range(dims[0]):
            row, offset = _reshape(offset, dims[1:])
            rows.append(row)
        return rows, offset

    reshaped, _offset = _reshape(0, list(shape))
    return reshaped


def _tensor_dtype_name(dtype: Any) -> str:
    name = str(dtype).removeprefix("torch.").removeprefix("numpy.")
    if name.startswith("float") or name in {"bfloat16", "double"}:
        return "float32"
    if name.startswith("int") or name.startswith("uint") or name == "long":
        return "int64"
    return name


def _numpy_dtype(dtype: str):
    import numpy as np

    if dtype.startswith("float") or dtype in {"bfloat16", "double"}:
        return np.float32
    if dtype.startswith("int") or dtype.startswith("uint") or dtype == "long":
        return np.int64
    raise ValueError(f"Unsupported TensorData dtype for numpy conversion: {dtype}")


def _torch_tensor_dtype(dtype: str):
    import torch

    if dtype.startswith("float") or dtype in {"bfloat16", "double"}:
        return torch.float32
    if dtype.startswith("int") or dtype.startswith("uint") or dtype == "long":
        return torch.int64
    raise ValueError(f"Unsupported TensorData dtype for torch conversion: {dtype}")


class RayObjectFuture(Generic[T]):
    """Small future wrapper around a Ray ObjectRef."""

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

    async def result_async(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

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

    async def result_async(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def get(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def __await__(self):
        async def _resolve():
            return self.result()

        return _resolve().__await__()


class FutureValueProxy(Generic[T]):
    """Resolved value that still accepts `.result()` for older local examples."""

    def __init__(self, value: T):
        object.__setattr__(self, "_value", value)

    def result(self, timeout: float | None = None) -> T:
        del timeout
        return object.__getattribute__(self, "_value")

    async def result_async(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def get(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def __await__(self):
        async def _resolve():
            return self.result()

        return _resolve().__await__()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.result(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.result(), name, value)

    def __call__(self, *args, **kwargs):
        return self.result()(*args, **kwargs)

    def __iter__(self):
        return iter(self.result())

    def __len__(self) -> int:
        return len(self.result())  # type: ignore[arg-type]

    def __getitem__(self, key):
        return self.result()[key]  # type: ignore[index]

    def __bool__(self) -> bool:
        return bool(self.result())

    def __str__(self) -> str:
        return str(self.result())

    def __repr__(self) -> str:
        return repr(self.result())

    def __eq__(self, other: Any) -> bool:
        return self.result() == other


def future_from(value: Any) -> RayObjectFuture[Any] | ImmediateFuture[Any]:
    """Wrap Ray object refs and synchronous values behind one future protocol."""

    if hasattr(value, "hex") and value.__class__.__name__ == "ObjectRef":
        return RayObjectFuture(value)
    return ImmediateFuture(value)


class AsyncMethodFuture(Generic[T]):
    """Future returned by async training client methods before completion."""

    def __init__(
        self,
        future: RayObjectFuture[T] | ImmediateFuture[T] | None = None,
        *,
        submit_sync: Callable[[], RayObjectFuture[T] | ImmediateFuture[T]] | None = None,
        submit_async: Callable[[], Awaitable[Any]] | None = None,
    ):
        self._future = future
        self._submit_sync = submit_sync
        self._submit_async = submit_async

    def _ensure_sync_future(self) -> RayObjectFuture[T] | ImmediateFuture[T]:
        if self._future is None:
            if self._submit_sync is None:
                raise RuntimeError("This async future can only be resolved from an async context.")
            self._future = self._submit_sync()
        return self._future

    async def _ensure_async_future(self) -> RayObjectFuture[T] | ImmediateFuture[T]:
        if self._future is None:
            if self._submit_async is None:
                return self._ensure_sync_future()
            self._future = future_from(await self._submit_async())
        return self._future

    def result(self, timeout: float | None = None) -> T:
        return self._ensure_sync_future().result(timeout=timeout)

    async def result_async(self, timeout: float | None = None) -> T:
        return await (await self._ensure_async_future()).result_async(timeout=timeout)

    def get(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def __await__(self):
        async def _submitted():
            await self._ensure_async_future()
            return self

        return _submitted().__await__()


def async_method_future(value: RayObjectFuture[T] | ImmediateFuture[T]) -> AsyncMethodFuture[T]:
    return AsyncMethodFuture(value)


@dataclass(slots=True)
class EncodedTextChunk:
    tokens: Sequence[int]
    type: Literal["encoded_text"] = "encoded_text"

    def __post_init__(self) -> None:
        self.tokens = [int(token) for token in self.tokens]

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass(slots=True)
class ImageChunk:
    data: bytes | str
    format: Literal["png", "jpeg"]
    expected_tokens: int | None = None
    type: Literal["image"] = "image"

    @property
    def length(self) -> int:
        if self.expected_tokens is None:
            raise ValueError("ImageChunk expected_tokens needs to be set in order to compute the length")
        return self.expected_tokens


@dataclass(slots=True)
class ImageAssetPointerChunk:
    format: Literal["png", "jpeg"]
    location: str
    expected_tokens: int | None = None
    type: Literal["image_asset_pointer"] = "image_asset_pointer"

    @property
    def length(self) -> int:
        if self.expected_tokens is None:
            raise ValueError("ImageAssetPointerChunk expected_tokens needs to be set in order to compute the length")
        return self.expected_tokens


ModelInputChunk = EncodedTextChunk | ImageChunk | ImageAssetPointerChunk


@dataclass(slots=True, init=False)
class ModelInput:
    """Tinker-compatible model input made of encoded text and optional images."""

    chunks: list[ModelInputChunk]

    def __init__(
        self,
        chunks: Sequence[ModelInputChunk] | None = None,
        *,
        tokens: Iterable[int] | None = None,
    ) -> None:
        if chunks is not None and tokens is not None:
            raise ValueError("Provide either chunks or tokens, not both.")
        if chunks is None:
            chunks = [] if tokens is None else [EncodedTextChunk(tokens=list(tokens))]
        self.chunks = list(chunks)

    @classmethod
    def from_ints(cls, tokens: Iterable[int]) -> "ModelInput":
        return cls(chunks=[EncodedTextChunk(tokens=list(tokens))])

    @classmethod
    def empty(cls) -> "ModelInput":
        return cls(chunks=[])

    def to_ints(self) -> list[int]:
        if not all(isinstance(chunk, EncodedTextChunk) for chunk in self.chunks):
            raise ValueError(
                "to_ints only supports ModelInput with EncodedTextChunks; "
                f"got {[type(chunk).__name__ for chunk in self.chunks]}"
            )
        return [int(token) for chunk in self.chunks for token in chunk.tokens]

    @property
    def tokens(self) -> list[int]:
        return self.to_ints()

    @property
    def length(self) -> int:
        return sum(chunk.length for chunk in self.chunks)

    def append(self, chunk: "ModelInput | ModelInputChunk | Iterable[int] | int") -> "ModelInput":
        if isinstance(chunk, ModelInput):
            return ModelInput(chunks=self.chunks + chunk.chunks)
        if isinstance(chunk, (EncodedTextChunk, ImageChunk, ImageAssetPointerChunk)):
            return ModelInput(chunks=self.chunks + [chunk])
        if isinstance(chunk, int):
            return self.append_int(chunk)
        return ModelInput(chunks=self.chunks + [EncodedTextChunk(tokens=list(chunk))])

    def append_int(self, token: int) -> "ModelInput":
        return self.append(EncodedTextChunk(tokens=[token]))


@dataclass(slots=True)
class TensorData:
    """Small serializable tensor container compatible with Tinker-style inputs."""

    data: Any
    dtype: str
    shape: list[int] | None = None
    sparse_crow_indices: list[int] | None = None
    sparse_col_indices: list[int] | None = None

    def __post_init__(self) -> None:
        if self.shape is None and hasattr(self.data, "shape"):
            self.shape = list(self.data.shape)
        self.data = _flatten_nested(self.data)
        self.dtype = _tensor_dtype_name(self.dtype)

    @classmethod
    def from_numpy(cls, array: Any) -> "TensorData":
        return cls(
            data=array.flatten().tolist() if hasattr(array, "flatten") else _flatten_nested(array),
            dtype=_tensor_dtype_name(getattr(array, "dtype", "float32")),
            shape=list(getattr(array, "shape", [])) or None,
        )

    @classmethod
    def from_torch(cls, tensor: Any) -> "TensorData":
        return cls(
            data=tensor.detach().cpu().flatten().tolist(),
            dtype=_tensor_dtype_name(tensor.dtype),
            shape=list(tensor.shape),
        )

    @classmethod
    def from_torch_sparse(cls, tensor: Any) -> "TensorData":
        return cls.from_torch(tensor)

    def to_numpy(self):
        import numpy as np

        array = np.array(self.data, dtype=_numpy_dtype(self.dtype))
        if self.shape is not None:
            array = array.reshape(self.shape)
        return array

    def to_torch(self):
        import torch

        tensor = torch.tensor(self.data, dtype=_torch_tensor_dtype(self.dtype))
        if self.shape is not None:
            tensor = tensor.reshape(self.shape)
        return tensor

    def tolist(self) -> Any:
        return _reshape_flat(self.data, self.shape)


def _convert_tensor_data(value: Any) -> Any:
    if isinstance(value, (TensorData, ModelInput)):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        return TensorData.from_torch(value)
    if hasattr(value, "tolist") and hasattr(value, "shape") and hasattr(value, "dtype"):
        return TensorData.from_numpy(value)
    if isinstance(value, dict):
        return {key: _convert_tensor_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_convert_tensor_data(item) for item in value]
    if isinstance(value, tuple):
        return [_convert_tensor_data(item) for item in value]
    return value


@dataclass(slots=True)
class Datum:
    """One training/evaluation datum."""

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
        self._tokens_list = list(tokens) if tokens is not None else (list(_tokens_list) if _tokens_list is not None else None)
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

    def result(self, timeout: float | None = None) -> "SampleResponse":
        del timeout
        return self

    async def result_async(self, timeout: float | None = None) -> "SampleResponse":
        return self.result(timeout=timeout)

    def get(self, timeout: float | None = None) -> "SampleResponse":
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


SaveWeightsResponse = CheckpointRef


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
    checkpoint: CheckpointRef | None = None


@dataclass(slots=True)
class ModelData:
    model_name: str
    model_id: str | None = None
    lora_rank: int | None = None
    tokenizer_id: str | None = None


@dataclass(slots=True)
class TrainingClientInfo:
    session_id: str
    base_model: str
    lora_rank: int
    step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_data(self) -> ModelData:
        return ModelData(model_name=self.base_model, model_id=self.session_id, lora_rank=self.lora_rank)


GetInfoResponse = TrainingClientInfo


@dataclass(slots=True)
class LoraConfig:
    rank: int = 32
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True


@dataclass(slots=True)
class SupportedModel:
    name: str
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GetServerCapabilitiesResponse:
    supported_models: list[str]
    supports_lora: bool = True
    supports_custom_loss: bool = True
    supports_multi_sampler: bool = True
    supports_multi_trainer: bool = False
    max_sampler_replicas: int = 1
    max_batch_size: int | None = None
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Cursor:
    next: str | None = None
    previous: str | None = None


@dataclass(slots=True)
class CheckpointArchiveUrlResponse:
    url: str
    expires_at: str | None = None


@dataclass(slots=True)
class ParsedCheckpointTinkerPath:
    run_id: str
    checkpoint_type: str
    name: str


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
