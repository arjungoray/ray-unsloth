"""Runtime configuration for Ray and Unsloth."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from ray_unsloth.errors import ConfigurationError

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

_WARNED_LEGACY_MODAL_SWITCH = False


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
    fast_inference: bool | str = "auto"
    gpu_memory_utilization: float = 0.85
    trust_remote_code: bool = True
    device_map: Any | None = None
    attn_implementation: str | None = None


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
        name: str = "<name>",
    ) -> ModelRuntimeConfig:
        return cls(
            model=_replace_dataclass(default_model, data.get("model", {}), path=f"model_configs.{name}.model"),
            lora=_replace_dataclass(default_lora, data.get("lora", {}), path=f"model_configs.{name}.lora"),
        )


@dataclass(slots=True)
class ResourceConfig:
    trainer_num_gpus: float = 1.0
    trainer_num_cpus: float = 1.0
    trainer_replicas: int = 1
    sampler_num_gpus: float = 1.0
    sampler_num_cpus: float = 1.0
    sampler_replicas: int = 1
    placement_strategy: str = "PACK"


@dataclass(slots=True)
class SpeedConfig:
    profile: str = "quality"
    padding_free: str | bool = "auto"
    sample_packing: str | bool = "auto"
    optimizer: str = "adamw_8bit"
    vllm_standby: str | bool = "auto"
    flash_attention_2: str | bool = "auto"
    live_policy_sampling: bool = True

    def __post_init__(self) -> None:
        if self.profile not in {"quality", "throughput"}:
            raise ConfigurationError(
                "speed.profile must be 'quality' or 'throughput'.",
                code="RU-1001",
                hint="Choose speed.profile: quality or throughput.",
            )
        if self.optimizer not in {"adamw_8bit", "paged_adamw_8bit", "adamw_torch"}:
            raise ConfigurationError(
                "speed.optimizer must be 'adamw_8bit', 'paged_adamw_8bit', or 'adamw_torch'.",
                code="RU-1002",
                hint="Pick one of the supported optimizer names.",
            )
        for field_name in ("padding_free", "sample_packing", "vllm_standby", "flash_attention_2"):
            value = getattr(self, field_name)
            if value not in {"auto", True, False}:
                raise ConfigurationError(
                    f"speed.{field_name} must be 'auto', true, or false.",
                    code="RU-1003",
                    hint="Use auto, true, or false for the speed toggle.",
                )


@dataclass(slots=True)
class DistributedConfig:
    enabled: bool = False
    mode: str | None = None
    num_nodes: int = 1
    gpus_per_node: int = 1
    backend: str = "nccl"
    placement_strategy: str = "STRICT_PACK"

    def validate(self) -> None:
        if self.mode is not None:
            self.enabled = True
        if not self.enabled and self.mode is None:
            return
        if self.mode != "ddp":
            raise ConfigurationError(
                "distributed.mode must be 'ddp' when distributed training is enabled.",
                code="RU-1004",
                hint="Set distributed.mode to ddp or disable distributed training.",
            )
        if self.num_nodes != 1:
            raise ConfigurationError(
                "Phase 1 distributed training only supports distributed.num_nodes == 1.",
                code="RU-1005",
                hint="Keep distributed.num_nodes at 1 for the current runtime.",
            )
        if self.gpus_per_node < 1:
            raise ConfigurationError(
                "distributed.gpus_per_node must be at least 1.",
                code="RU-1006",
                hint="Use at least one GPU per node when enabling distributed training.",
            )


@dataclass(slots=True)
class ModalConfig:
    enabled: bool = False
    app_name: str = "ray-unsloth"
    gpu: str = "L4"
    timeout: int = 1800
    scaledown_window: int = 300
    max_inputs: int | None = None
    trainer_pool_key: str | None = None
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
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    modal: ModalConfig = field(default_factory=ModalConfig)
    checkpoint_root: str = "checkpoints"
    supported_models: list[str] = field(default_factory=list)
    provider: str | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)
    plugins: list[str] = field(default_factory=list)
    scribe: dict[str, Any] = field(default_factory=dict)
    run_name: str | None = None
    tracking: bool = True
    tracking_root: str | None = None

    def __post_init__(self) -> None:
        self.distributed.validate()
        global _WARNED_LEGACY_MODAL_SWITCH
        if self.provider is None and self.modal.enabled and not _WARNED_LEGACY_MODAL_SWITCH:
            warnings.warn(
                "modal.enabled is deprecated; set provider: modal instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            _WARNED_LEGACY_MODAL_SWITCH = True
        if self.provider is not None and self.modal.enabled and self.provider not in ("modal",):
            raise ConfigurationError(
                f"Conflicting runtime selection: provider is '{self.provider}' but modal.enabled is true. "
                "Set provider: modal, or drop modal.enabled (it is the legacy switch).",
                code="RU-1007",
                hint="Prefer provider: modal and remove the legacy modal.enabled switch.",
            )

    @classmethod
    def from_file(cls, path: str | Path) -> RuntimeConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RuntimeConfig:
        data = data or {}
        model_data = data.get("model", {})
        default_model_config: str | None = None
        if isinstance(model_data, str):
            default_model_config = model_data
            model_data = {}
        else:
            model_data = dict(model_data)
            default_model_config = model_data.pop("config", None)
        model = _build_dataclass(ModelConfig, model_data, path="model")
        lora = _build_dataclass(LoRAConfig, data.get("lora", {}), path="lora")
        model_configs = {}
        for name, config in data.get("model_configs", {}).items():
            if not isinstance(config, dict):
                raise ConfigurationError(
                    f"model_configs.{name} must be a mapping with optional model and lora sections.",
                    code="RU-1008",
                    hint="Each model_configs entry must be a YAML mapping.",
                )
            model_configs[name] = ModelRuntimeConfig.from_dict(
                config,
                default_model=model,
                default_lora=lora,
                name=name,
            )
        if default_model_config is not None:
            if default_model_config not in model_configs:
                available = _format_available_model_configs(model_configs)
                raise ConfigurationError(
                    f"model.config '{default_model_config}' does not match a key in model_configs. "
                    f"Available model configs: {available}. Set model.config to one of these aliases "
                    "or remove model.config to use model.base_model.",
                    code="RU-1009",
                    hint="Set model.config to one of the keys listed in model_configs.",
                )
            selected = model_configs[default_model_config]
            model = selected.model
            lora = selected.lora
        return cls(
            ray=_build_dataclass(RayConfig, data.get("ray", {}), path="ray"),
            model=model,
            lora=lora,
            default_model_config=default_model_config,
            model_configs=model_configs,
            resources=_build_dataclass(ResourceConfig, data.get("resources", {}), path="resources"),
            speed=_build_dataclass(SpeedConfig, data.get("speed", {}), path="speed"),
            distributed=_build_dataclass(DistributedConfig, data.get("distributed", {}), path="distributed"),
            modal=_build_dataclass(ModalConfig, data.get("modal", {}), path="modal"),
            checkpoint_root=data.get("checkpoint_root", "checkpoints"),
            supported_models=list(data.get("supported_models", [])),
            provider=data.get("provider"),
            provider_options=dict(data.get("provider_options", {})),
            plugins=list(data.get("plugins", [])),
            scribe=dict(data.get("scribe", {})),
            run_name=data.get("run_name"),
            tracking=bool(data.get("tracking", True)),
            tracking_root=data.get("tracking_root"),
        )

    @property
    def store_root(self) -> str:
        """Where the client-side run store lives.

        Defaults to ``checkpoint_root``, but ``tracking_root`` overrides it for
        topologies where ``checkpoint_root`` is a remote volume path (e.g. the
        Modal ``/checkpoints`` mount) that is not writable on the client.

        When neither is configured (checkpoint_root is still the bare default),
        the ``RAY_UNSLOTH_DEFAULT_STORE_ROOT`` environment variable relocates
        the store — used by the test suite to keep default-config clients from
        writing into the working tree.
        """
        if self.tracking_root:
            return self.tracking_root
        if self.checkpoint_root == "checkpoints":
            import os

            override = os.environ.get("RAY_UNSLOTH_DEFAULT_STORE_ROOT")
            if override:
                return override
        return self.checkpoint_root

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

    def validate(self) -> list[Any]:
        """Return non-throwing validation issues for CLI/UI workflows."""
        from ray_unsloth.providers import get_provider, resolve_provider_name
        from ray_unsloth.providers.base import ValidationIssue, estimate_gpu_fit

        issues: list[Any] = []
        provider_name = resolve_provider_name(self)
        try:
            provider = get_provider(provider_name)
        except Exception as exc:
            return [
                ValidationIssue(
                    severity="error",
                    path="provider",
                    message=f"Unknown or invalid provider '{provider_name}': {exc}",
                )
            ]
        issues.extend(provider.validate(self))
        gpu = (self.provider_options or {}).get("gpu", self.modal.gpu)
        fit = estimate_gpu_fit(self.model, self.lora, gpu=gpu)
        if fit.fits is False:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path="model",
                    message=(
                        f"Estimated memory for {self.model.base_model} is "
                        f"{fit.estimated_required_gb:.1f} GB on {gpu}; this likely will not fit."
                    ),
                    hint="Choose a larger GPU, shorter sequence length, or smaller model.",
                )
            )
        return issues


def _build_dataclass(cls, data: dict[str, Any], *, path: str):
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"{path} must be a mapping.",
            code="RU-1010",
            hint="Use a YAML mapping for this section.",
        )
    field_names = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - field_names)
    if unknown:
        raise ConfigurationError(
            f"Unknown config field(s) in {path}: {unknown}. Valid fields: {sorted(field_names)}.",
            code="RU-1010",
            hint="Remove the unknown keys or rename them to match the config dataclass.",
        )
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigurationError(
            f"Invalid value for {path}: {exc}",
            code="RU-1010",
            hint="Check the field types and nested mappings in the YAML config.",
        ) from exc


def _replace_dataclass(instance, updates: dict[str, Any], *, path: str):
    values = asdict(instance)
    field_names = set(values)
    unknown = sorted(set(updates) - field_names)
    if unknown:
        raise ConfigurationError(
            f"Unknown config field(s) in {path}: {unknown}. Valid fields: {sorted(field_names)}.",
            code="RU-1010",
            hint="Remove the unknown keys or rename them to match the config dataclass.",
        )
    values.update(updates)
    return type(instance)(**values)


def _format_available_model_configs(model_configs: dict[str, ModelRuntimeConfig]) -> str:
    if not model_configs:
        return "none"
    entries = [f"{name} -> {config.model.base_model}" for name, config in model_configs.items()]
    return ", ".join(entries)


def load_config(config: str | Path | RuntimeConfig | dict[str, Any] | None) -> RuntimeConfig:
    if config is None:
        return RuntimeConfig()
    if isinstance(config, RuntimeConfig):
        loaded = config
    elif isinstance(config, (str, Path)):
        loaded = RuntimeConfig.from_file(config)
    else:
        loaded = RuntimeConfig.from_dict(config)
    if loaded.plugins:
        from ray_unsloth.plugins import load_config_plugins

        load_config_plugins(loaded.plugins)
    return loaded


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
