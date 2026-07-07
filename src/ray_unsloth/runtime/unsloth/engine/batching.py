"""Batch construction helpers for the Unsloth engine."""

from __future__ import annotations

from .core import UnslothEngine, _BatchPlan

_batch_from_data = UnslothEngine._batch_from_data

__all__ = ["_BatchPlan", "_batch_from_data"]
