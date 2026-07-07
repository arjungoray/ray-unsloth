from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
class SamplerDownloadResponse:
    path: str
    archive_path: str
    archive_relpath: str
    token: str
    expires_at: int
    url: str | None = None
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
    max_concurrent_trainers: int = 1
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


__all__ = [
    "Checkpoint",
    "CheckpointArchiveUrlResponse",
    "CheckpointRef",
    "CheckpointsListResponse",
    "Cursor",
    "GetInfoResponse",
    "GetServerCapabilitiesResponse",
    "LoraConfig",
    "ModelData",
    "ParsedCheckpointTinkerPath",
    "SamplerDownloadResponse",
    "SaveWeightsForSamplerResponse",
    "SaveWeightsResponse",
    "SupportedModel",
    "TrainingClientInfo",
    "TrainingRun",
    "TrainingRunsResponse",
    "WeightsInfoResponse",
]
