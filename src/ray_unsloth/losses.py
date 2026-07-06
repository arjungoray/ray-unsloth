"""Loss function registry.

Built-in losses (``cross_entropy``, ``importance_sampling``, ``ppo``,
``cispo``) are described by :class:`LossSpec` entries in
``ray_unsloth.plugins.losses``. The engine dispatches through this registry,
so registering a new policy-gradient loss makes it runnable end-to-end
without modifying engine code:

    from ray_unsloth.losses import LossSpec, register_loss

    def grpo_token_loss(*, ratio, advantages, current_logprobs, config):
        clipped = ratio.clamp(1 - config["clip_eps"], 1 + config["clip_eps"])
        import torch
        return -torch.minimum(ratio * advantages, clipped * advantages)

    register_loss(LossSpec(
        name="grpo_clip",
        kind="policy_gradient",
        description="GRPO-style clipped surrogate",
        required_inputs=("target_tokens", "logprobs", "advantages"),
        config_defaults={"clip_eps": 0.2},
        token_loss=grpo_token_loss,
    ))

A ``policy_gradient`` loss's ``token_loss`` receives torch tensors selected
at trainable positions (``ratio = exp(current - old)``) plus the merged
config dict, and returns per-token losses which the engine sums.

Losses with pairwise or otherwise non-token-parallel structure (DPO, KTO)
are expressed through ``TrainingClient.forward_backward_custom`` instead —
see ``examples/sample_plugin`` for a working DPO-style example.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from ray_unsloth.errors import UnsupportedLossError
from ray_unsloth.plugins import losses as _registry

TokenLossFn = Callable[..., Any]


@dataclass(frozen=True)
class LossSpec:
    """Metadata + (for policy-gradient losses) the token-loss implementation."""

    name: str
    kind: str  # "supervised" | "policy_gradient"
    description: str
    required_inputs: tuple[str, ...] = ()
    optional_inputs: tuple[str, ...] = ()
    config_defaults: dict[str, float] = field(default_factory=dict)
    token_loss: TokenLossFn | None = None

    def merged_config(self, overrides: dict[str, Any] | None) -> dict[str, Any]:
        config = dict(self.config_defaults)
        if overrides:
            config.update(overrides)
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "required_inputs": list(self.required_inputs),
            "optional_inputs": list(self.optional_inputs),
            "config_defaults": dict(self.config_defaults),
        }


def register_loss(spec: LossSpec, *, replace: bool = False) -> LossSpec:
    _registry.register(spec.name, spec, description=spec.description, replace=replace)
    return spec


def get_loss(name: str) -> LossSpec:
    try:
        return cast(LossSpec, _registry.get(name))
    except Exception:
        raise UnsupportedLossError(
            f"Unsupported loss '{name}'. Registered losses: {', '.join(_registry.names())}."
        ) from None


def list_losses() -> list[LossSpec]:
    return [spec for _name, spec in _registry.items()]


def loss_names() -> list[str]:
    return _registry.names()


def validate_datum_inputs(spec: LossSpec, loss_fn_inputs: dict[str, Any], *, datum_index: int = 0) -> None:
    """Raise a descriptive error when a datum is missing required loss inputs."""
    missing = [key for key in spec.required_inputs if key not in loss_fn_inputs]
    if missing:
        raise UnsupportedLossError(
            f"Datum {datum_index} is missing required loss_fn_inputs for '{spec.name}': "
            f"{missing}. Required: {list(spec.required_inputs)}; "
            f"optional: {list(spec.optional_inputs)}."
        )


# ---------------------------------------------------------------------------
# Built-in loss implementations. Token-loss math mirrors the engine's
# original dispatch exactly (see runtime/unsloth/engine.py history).
# ---------------------------------------------------------------------------


def _importance_sampling_token_loss(*, ratio: Any, advantages: Any, current_logprobs: Any, config: dict[str, Any]) -> Any:
    del current_logprobs, config
    return -ratio * advantages


def _ppo_token_loss(*, ratio: Any, advantages: Any, current_logprobs: Any, config: dict[str, Any]) -> Any:
    del current_logprobs
    import torch

    clipped_ratio = ratio.clamp(config["clip_low_threshold"], config["clip_high_threshold"])
    return -torch.minimum(ratio * advantages, clipped_ratio * advantages)


def _cispo_token_loss(*, ratio: Any, advantages: Any, current_logprobs: Any, config: dict[str, Any]) -> Any:
    clipped_ratio = ratio.detach().clamp(config["clip_low_threshold"], config["clip_high_threshold"])
    return -clipped_ratio * advantages * current_logprobs


CROSS_ENTROPY = LossSpec(
    name="cross_entropy",
    kind="supervised",
    description="Token-level negative log-likelihood with optional per-token weights.",
    required_inputs=(),
    optional_inputs=("target_tokens", "labels", "weights"),
)

IMPORTANCE_SAMPLING = LossSpec(
    name="importance_sampling",
    kind="policy_gradient",
    description="REINFORCE with importance-sampling ratios (exp(new - old) * advantage).",
    required_inputs=("target_tokens", "logprobs", "advantages"),
    optional_inputs=("weights",),
    token_loss=_importance_sampling_token_loss,
)

PPO = LossSpec(
    name="ppo",
    kind="policy_gradient",
    description="PPO clipped surrogate objective.",
    required_inputs=("target_tokens", "logprobs", "advantages"),
    optional_inputs=("weights",),
    config_defaults={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
    token_loss=_ppo_token_loss,
)

CISPO = LossSpec(
    name="cispo",
    kind="policy_gradient",
    description="Clipped importance-sampling policy optimization (detached clipped ratio × logprob).",
    required_inputs=("target_tokens", "logprobs", "advantages"),
    optional_inputs=("weights",),
    config_defaults={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
    token_loss=_cispo_token_loss,
)

for _spec in (CROSS_ENTROPY, IMPORTANCE_SAMPLING, PPO, CISPO):
    if _spec.name not in _registry:
        register_loss(_spec)
