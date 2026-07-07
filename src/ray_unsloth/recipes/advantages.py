"""Advantage and weight helpers for group-relative training."""

from __future__ import annotations

from math import sqrt


def group_relative(rewards: list[float], *, normalize_std: bool = False) -> list[float]:
    """Center rewards around their mean and optionally z-normalize them."""

    if not rewards:
        return []
    mean = sum(float(reward) for reward in rewards) / len(rewards)
    centered = [float(reward) - mean for reward in rewards]
    if not normalize_std:
        return centered
    variance = sum(value * value for value in centered) / len(centered)
    std = sqrt(variance)
    if std == 0.0:
        return [0.0 for _ in centered]
    return [value / std for value in centered]


def drop_uniform_groups(groups: list[list[float]], threshold: float = 0.05) -> list[list[float]]:
    """Return only groups whose reward spread exceeds ``threshold``."""

    kept: list[list[float]] = []
    for group in groups:
        if not group:
            continue
        if max(group) - min(group) > threshold:
            kept.append(group)
    return kept


def length_normalized_weights(completion_len: int) -> float:
    """Return the per-token weight that gives a completion unit total mass."""

    if completion_len <= 0:
        raise ValueError("completion_len must be positive")
    return 1.0 / float(completion_len)


__all__ = [
    "drop_uniform_groups",
    "group_relative",
    "length_normalized_weights",
]
