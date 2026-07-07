"""Rollout collection and policy-datum conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ray_unsloth import Datum, ModelInput, TensorData
from ray_unsloth.clients._remote import resolve
from ray_unsloth.recipes.advantages import length_normalized_weights


@dataclass(slots=True)
class Rollout:
    """One sampled completion plus the tokens and token logprobs used for policy learning."""

    tokens: list[int]
    text: str
    logprobs: list[float]


def collect_group(
    sampling_client: Any,
    prompt: ModelInput,
    *,
    group_size: int,
    sampling_params: Any,
) -> list[Rollout]:
    """Sample a group of completions for one prompt."""

    response = resolve(
        sampling_client.sample(
            prompt,
            num_samples=group_size,
            sampling_params=sampling_params,
        )
    )
    rollouts: list[Rollout] = []
    prompt_tokens = prompt.to_ints()
    for sequence in response.sequences:
        tokens = [int(token) for token in sequence.tokens]
        text = sequence.text or ""
        logprobs = _completion_logprobs(
            sampling_client=sampling_client,
            prompt_tokens=prompt_tokens,
            completion_tokens=tokens,
            sequence_logprobs=sequence.logprobs,
        )
        rollouts.append(Rollout(tokens=tokens, text=text, logprobs=logprobs))
    return rollouts


def rollout_to_datum(prompt_tokens: list[int], rollout: Rollout, advantage: float) -> Datum:
    """Convert a rollout into the exact policy-loss datum contract used by the engine."""

    if not rollout.tokens:
        raise ValueError("rollout.tokens must not be empty")
    completion_tokens = [int(token) for token in rollout.tokens]
    prompt_padding = max(len(prompt_tokens) - 1, 0)
    model_tokens = prompt_tokens + completion_tokens[:-1] if prompt_tokens else completion_tokens
    padded_logprobs = _pad_values(rollout.logprobs, len(completion_tokens))
    target_tokens = [-100] * prompt_padding + completion_tokens
    logprobs = [0.0] * prompt_padding + padded_logprobs
    advantages = [0.0] * prompt_padding + [float(advantage)] * len(completion_tokens)
    weights = [0.0] * prompt_padding + [length_normalized_weights(len(completion_tokens))] * len(completion_tokens)
    return Datum(
        model_input=ModelInput.from_ints(model_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData(data=target_tokens, dtype="int64", shape=[len(target_tokens)]),
            "logprobs": TensorData(data=logprobs, dtype="float32", shape=[len(logprobs)]),
            "advantages": TensorData(data=advantages, dtype="float32", shape=[len(advantages)]),
            "weights": TensorData(data=weights, dtype="float32", shape=[len(weights)]),
        },
    )


def _completion_logprobs(
    *,
    sampling_client: Any,
    prompt_tokens: list[int],
    completion_tokens: list[int],
    sequence_logprobs: list[float | None] | None,
) -> list[float]:
    if sequence_logprobs is not None and len(sequence_logprobs) >= len(completion_tokens):
        return _pad_values(
            [0.0 if value is None else float(value) for value in sequence_logprobs], len(completion_tokens)
        )
    prompt_prefix = prompt_tokens if prompt_tokens else [0]
    full_tokens = prompt_prefix + completion_tokens
    computed = resolve(sampling_client.compute_logprobs(ModelInput.from_ints(full_tokens)))
    values = [0.0 if value is None else float(value) for value in computed]
    return _pad_values(values[len(prompt_prefix) :], len(completion_tokens))


def _pad_values(values: list[float], target_len: int) -> list[float]:
    padded = list(values[:target_len])
    if len(padded) < target_len:
        padded.extend([0.0] * (target_len - len(padded)))
    return padded


__all__ = [
    "Rollout",
    "collect_group",
    "rollout_to_datum",
]
