"""Loss prep and dispatch helpers for the Unsloth engine."""

from __future__ import annotations

from .core import UnslothEngine

_ensure_optimizer = UnslothEngine._ensure_optimizer
_batch_from_data = UnslothEngine._batch_from_data
_loss_tensor_data = UnslothEngine._loss_tensor_data
_label_rows_from_data = UnslothEngine._label_rows_from_data
_loss_tokens = UnslothEngine._loss_tokens
_model_forward = UnslothEngine._model_forward
_weighted_positions_to_keep = UnslothEngine._weighted_positions_to_keep
_logit_position = UnslothEngine._logit_position
_cross_entropy_loss = UnslothEngine._cross_entropy_loss
_cross_entropy_loss_for_plan = UnslothEngine._cross_entropy_loss_for_plan
_policy_loss = UnslothEngine._policy_loss
_policy_loss_for_plan = UnslothEngine._policy_loss_for_plan
_loss_fn_outputs = UnslothEngine._loss_fn_outputs
forward = UnslothEngine.forward
forward_backward = UnslothEngine.forward_backward
forward_backward_custom = UnslothEngine.forward_backward_custom
optim_step = UnslothEngine.optim_step

__all__ = [
    "_batch_from_data",
    "_cross_entropy_loss",
    "_cross_entropy_loss_for_plan",
    "_ensure_optimizer",
    "_label_rows_from_data",
    "_logit_position",
    "_loss_fn_outputs",
    "_loss_tensor_data",
    "_loss_tokens",
    "_model_forward",
    "_policy_loss",
    "_policy_loss_for_plan",
    "_weighted_positions_to_keep",
    "forward",
    "forward_backward",
    "forward_backward_custom",
    "optim_step",
]
