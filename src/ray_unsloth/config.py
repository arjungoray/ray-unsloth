"""Runtime configuration for Ray and Unsloth."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
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
    base_model: str = "unsloth/gemma-4-E2B-it"
    max_seq_length: int = 2048
    dtype: str = "bfloat16"
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.85
    trust_remote_code: bool = True


@dataclass(slots=True)
class LoRAConfig:
    rank: int = 32
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES))
    bias: str = "none"
    use_gradient_checkpointing: str | bool = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: dict[str, Any] | None = None


@dataclass(slots=True)
class ModelRuntimeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        default_model: ModelConfig,
        default_lora: LoRAConfig,
    ) -> "ModelRuntimeConfig":
        return cls(
            model=_replace_dataclass(default_model, data.get("model", {})),
            lora=_replace_dataclass(default_lora, data.get("lora", {})),
        )


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
    python_version: str = "3.11"


@dataclass(slots=True)
class RuntimeConfig:
    ray: RayConfig = field(default_factory=RayConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    default_model_config: str | None = None
    model_configs: dict[str, ModelRuntimeConfig] = field(default_factory=dict)
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
        model_data = data.get("model", {})
        default_model_config: str | None = None
        if isinstance(model_data, str):
            default_model_config = model_data
            model_data = {}
        else:
            model_data = dict(model_data)
            default_model_config = model_data.pop("config", None)
        model = ModelConfig(**model_data)
        lora = LoRAConfig(**data.get("lora", {}))
        model_configs = {
            name: ModelRuntimeConfig.from_dict(
                config,
                default_model=model,
                default_lora=lora,
            )
            for name, config in data.get("model_configs", {}).items()
        }
        if default_model_config is not None:
            if default_model_config not in model_configs:
                raise ValueError(f"Unknown model config: {default_model_config}")
            selected = model_configs[default_model_config]
            model = selected.model
            lora = selected.lora
        return cls(
            ray=RayConfig(**data.get("ray", {})),
            model=model,
            lora=lora,
            default_model_config=default_model_config,
            model_configs=model_configs,
            resources=ResourceConfig(**data.get("resources", {})),
            modal=ModalConfig(**data.get("modal", {})),
            checkpoint_root=data.get("checkpoint_root", "checkpoints"),
            supported_models=list(data.get("supported_models", [])),
        )

    def resolve_model_configs(self, base_model: str | None = None) -> tuple[ModelConfig, LoRAConfig]:
        """Return the model and LoRA config for a requested model name or alias."""

        if base_model is None:
            return self.model, self.lora
        runtime_config = self.model_configs.get(base_model)
        if runtime_config is not None:
            return runtime_config.model, runtime_config.lora
        for runtime_config in self.model_configs.values():
            if runtime_config.model.base_model == base_model:
                return runtime_config.model, runtime_config.lora
        return replace(self.model, base_model=base_model), self.lora

    def supported_model_names(self) -> list[str]:
        if self.supported_models:
            return self.supported_models
        if self.model_configs:
            return list(self.model_configs)
        return [self.model.base_model]


def _replace_dataclass(instance, updates: dict[str, Any]):
    values = asdict(instance)
    values.update(updates)
    return type(instance)(**values)


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
