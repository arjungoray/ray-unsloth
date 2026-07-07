from __future__ import annotations

from dataclasses import replace
from typing import Any

from ray_unsloth.config import LoRAConfig, ModelConfig, RuntimeConfig


def resolve_actor_configs(
    config: RuntimeConfig,
    *,
    base_model: str | None,
    lora_rank: int | None = None,
    seed: int | None = None,
    target_modules: list[str] | None = None,
) -> tuple[ModelConfig, LoRAConfig]:
    model_config, lora_config = config.resolve_model_configs(base_model)
    updates: dict[str, Any] = {}
    if lora_rank is not None:
        updates["rank"] = lora_rank
    if seed is not None:
        updates["random_state"] = seed
    if target_modules is not None:
        updates["target_modules"] = target_modules
    if updates:
        lora_config = replace(lora_config, **updates)
    return model_config, lora_config
