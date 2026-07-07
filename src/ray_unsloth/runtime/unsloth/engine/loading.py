"""Loading-related helpers for the Unsloth engine."""

from __future__ import annotations

from .core import (
    UnslothEngine,
    _flash_attention_2_available,
    _flash_attention_3_available,
    _module_available,
    _requires_transformers_5_for_qwen3_5,
    _torch_dtype,
)

_setup_distributed = UnslothEngine._setup_distributed
_wrap_distributed_model = UnslothEngine._wrap_distributed_model
_unwrap_model = UnslothEngine._unwrap_model
_generation_config_for_new_tokens = UnslothEngine._generation_config_for_new_tokens
_model_device = UnslothEngine._model_device
_load_model = UnslothEngine._load_model
_speed = UnslothEngine._speed
_effective_fast_inference = UnslothEngine._effective_fast_inference
_effective_vllm_standby = UnslothEngine._effective_vllm_standby
_effective_gpu_memory_utilization = UnslothEngine._effective_gpu_memory_utilization
_effective_attn_implementation = UnslothEngine._effective_attn_implementation
_gpu_compute_capability = UnslothEngine._gpu_compute_capability
_best_attn_backend = UnslothEngine._best_attn_backend
_configure_unsloth_environment = UnslothEngine._configure_unsloth_environment
_padding_free_requested = UnslothEngine._padding_free_requested
_padding_free_forced = UnslothEngine._padding_free_forced
_set_attention_implementation = UnslothEngine._set_attention_implementation
get_tokenizer = UnslothEngine.get_tokenizer
get_info = UnslothEngine.get_info
register_custom_loss = UnslothEngine.register_custom_loss

__all__ = [
    "_best_attn_backend",
    "_configure_unsloth_environment",
    "_effective_attn_implementation",
    "_effective_fast_inference",
    "_effective_gpu_memory_utilization",
    "_effective_vllm_standby",
    "_flash_attention_2_available",
    "_flash_attention_3_available",
    "_gpu_compute_capability",
    "_load_model",
    "_model_device",
    "_module_available",
    "_padding_free_forced",
    "_padding_free_requested",
    "_requires_transformers_5_for_qwen3_5",
    "_set_attention_implementation",
    "_setup_distributed",
    "_speed",
    "_torch_dtype",
    "_unwrap_model",
    "_wrap_distributed_model",
    "get_info",
    "get_tokenizer",
    "register_custom_loss",
]
