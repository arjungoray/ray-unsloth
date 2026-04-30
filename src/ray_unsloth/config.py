"""Runtime configuration for Ray and Unsloth."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

ATTN_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_LORA_TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]
UNEMBED_LORA_TARGET_MODULES = ["lm_head"]


@dataclass(slots=True)
class RayConfig:
    address: str | None = None
    namespace: str = "ray-unsloth"
    ignore_reinit_error: bool = True


@dataclass(slots=True)
class ModelConfig:
    base_model: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    dtype: str = "bfloat16"
    load_in_4bit: bool = True
    fast_inference: bool = False
    gpu_memory_utilization: float = 0.80
    trust_remote_code: bool = True


@dataclass(slots=True)
class LoRAConfig:
    rank: int = 32
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES))
    random_state: int = 3407
    use_rslora: bool = False


@dataclass(slots=True)
class ResourceConfig:
    trainer_num_gpus: float = 1.0
    trainer_num_cpus: float = 1.0
    sampler_num_gpus: float = 1.0
    sampler_num_cpus: float = 1.0
    sampler_replicas: int = 1
    placement_strategy: str = "PACK"


@dataclass(slots=True)
class ModalConfig:
    enabled: bool = False
    app_name: str = "ray-unsloth"
    gpu: str = "L4"
    timeout: int = 1800
    scaledown_window: int = 300
    volume_name: str = "ray-unsloth-checkpoints"
    volume_mount_path: str = "/checkpoints"
    python_version: str = "3.10"


@dataclass(slots=True)
class RuntimeConfig:
    ray: RayConfig = field(default_factory=RayConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    modal: ModalConfig = field(default_factory=ModalConfig)
    checkpoint_root: str = "checkpoints"
    supported_models: list[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "RuntimeConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RuntimeConfig":
        data = data or {}
        return cls(
            ray=RayConfig(**data.get("ray", {})),
            model=ModelConfig(**data.get("model", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            resources=ResourceConfig(**data.get("resources", {})),
            modal=ModalConfig(**data.get("modal", {})),
            checkpoint_root=data.get("checkpoint_root", "checkpoints"),
            supported_models=list(data.get("supported_models", [])),
        )


def load_config(config: str | Path | RuntimeConfig | dict[str, Any] | None) -> RuntimeConfig:
    if config is None:
        return RuntimeConfig()
    if isinstance(config, RuntimeConfig):
        return config
    if isinstance(config, (str, Path)):
        return RuntimeConfig.from_file(config)
    return RuntimeConfig.from_dict(config)


def lora_target_modules_for_flags(
    *,
    train_mlp: bool = True,
    train_attn: bool = True,
    train_unembed: bool = True,
) -> list[str]:
    modules: list[str] = []
    if train_attn:
        modules.extend(ATTN_LORA_TARGET_MODULES)
    if train_mlp:
        modules.extend(MLP_LORA_TARGET_MODULES)
    if train_unembed:
        modules.extend(UNEMBED_LORA_TARGET_MODULES)
    return modules
