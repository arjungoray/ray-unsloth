from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import prod
from typing import Any, Literal


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
    def from_ints(cls, tokens: Iterable[int]) -> ModelInput:
        return cls(chunks=[EncodedTextChunk(tokens=list(tokens))])

    @classmethod
    def empty(cls) -> ModelInput:
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

    def append(self, chunk: ModelInput | ModelInputChunk | Iterable[int] | int) -> ModelInput:
        if isinstance(chunk, ModelInput):
            return ModelInput(chunks=self.chunks + chunk.chunks)
        if isinstance(chunk, (EncodedTextChunk, ImageChunk, ImageAssetPointerChunk)):
            return ModelInput(chunks=[*self.chunks, chunk])
        if isinstance(chunk, int):
            return self.append_int(chunk)
        return ModelInput(chunks=[*self.chunks, EncodedTextChunk(tokens=list(chunk))])

    def append_int(self, token: int) -> ModelInput:
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
    def from_numpy(cls, array: Any) -> TensorData:
        return cls(
            data=array.flatten().tolist() if hasattr(array, "flatten") else _flatten_nested(array),
            dtype=_tensor_dtype_name(getattr(array, "dtype", "float32")),
            shape=list(getattr(array, "shape", [])) or None,
        )

    @classmethod
    def from_torch(cls, tensor: Any) -> TensorData:
        return cls(
            data=tensor.detach().cpu().flatten().tolist(),
            dtype=_tensor_dtype_name(tensor.dtype),
            shape=list(tensor.shape),
        )

    @classmethod
    def from_torch_sparse(cls, tensor: Any) -> TensorData:
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
    max_time: float | None = None
    logprobs_max_tokens: int | None = None


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


__all__ = [
    "AdamParams",
    "Datum",
    "EncodedTextChunk",
    "ImageAssetPointerChunk",
    "ImageChunk",
    "ModelInput",
    "ModelInputChunk",
    "SamplingParams",
    "TensorData",
]
