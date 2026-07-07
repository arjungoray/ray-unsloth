"""Reusable group-relative policy optimization rounds."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams
from ray_unsloth.clients._remote import resolve
from ray_unsloth.recipes.advantages import drop_uniform_groups, group_relative
from ray_unsloth.recipes.rewards import Rubric
from ray_unsloth.recipes.rollouts import collect_group, rollout_to_datum
from ray_unsloth.recipes.sft import _batch_token_normalized


@dataclass(slots=True)
class PromptSpec:
    """A prompt plus any reward-time context."""

    prompt_text: str
    prompt_tokens: list[int] | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GrpoConfig:
    """Configuration for a single GRPO round."""

    group_size: int = 8
    prompts_per_batch: int = 16
    batches_per_round: int = 12
    inner_epochs: int = 2
    loss_fn: str = "ppo"
    loss_fn_config: dict[str, Any] | None = None
    learning_rate: float = 5e-6
    anchor_weight: float = 0.05
    max_tokens: int = 320
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int = 0


@dataclass(slots=True)
class GrpoRoundReport:
    """Summary statistics for one GRPO round."""

    mean_reward: float
    per_term_means: dict[str, float]
    n_datums: int
    losses: list[float]
    n_scored_samples: int = 0


def grpo_round(
    training_client: Any,
    prompt_bank: list[PromptSpec],
    rubric: Rubric,
    config: GrpoConfig,
    *,
    anchor_datums: list[Datum] | None = None,
    recorder_log=None,
) -> GrpoRoundReport:
    """Run one GRPO round against a training client."""

    if not prompt_bank:
        raise ValueError("prompt_bank must not be empty")
    rng = random.Random(config.seed)
    tokenizer = resolve(training_client.get_tokenizer()) if hasattr(training_client, "get_tokenizer") else None
    sampling_client = training_client.create_live_sampling_client()
    sampling_params = SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=None,
    )

    scored_totals: list[float] = []
    per_term_totals: dict[str, float] = {}
    total_datums = 0
    losses: list[float] = []

    for _batch_index in range(config.batches_per_round):
        batch_datums: list[Datum] = []
        batch_prompts = [rng.choice(prompt_bank) for _ in range(config.prompts_per_batch)]
        for prompt_spec in batch_prompts:
            prompt_tokens = (
                list(prompt_spec.prompt_tokens)
                if prompt_spec.prompt_tokens is not None
                else _encode_prompt(
                    tokenizer,
                    prompt_spec.prompt_text,
                )
            )
            prompt_input = ModelInput.from_ints(prompt_tokens)
            rollouts = collect_group(
                sampling_client,
                prompt_input,
                group_size=config.group_size,
                sampling_params=sampling_params,
            )
            samples = [
                {
                    "prompt": prompt_spec.prompt_text,
                    "completion_text": rollout.text,
                    "completion_tokens": rollout.tokens,
                    "context": prompt_spec.context,
                }
                for rollout in rollouts
            ]
            breakdowns = rubric.score(samples)
            rewards = [breakdown.total for breakdown in breakdowns]
            for rollout, breakdown in zip(rollouts, breakdowns, strict=False):
                scored_totals.append(breakdown.total)
                for term_name, value in breakdown.terms.items():
                    per_term_totals[term_name] = per_term_totals.get(term_name, 0.0) + float(value)
                if recorder_log is not None:
                    recorder_log(
                        {
                            "prompt_text": prompt_spec.prompt_text,
                            "completion_text": rollout.text,
                            "completion_tokens": list(rollout.tokens),
                            "reward": breakdown.total,
                            "terms": dict(breakdown.terms),
                            "context": dict(prompt_spec.context),
                        }
                    )
            if not drop_uniform_groups([rewards], threshold=0.05):
                continue
            advantages = group_relative(rewards, normalize_std=False)
            for rollout, advantage in zip(rollouts, advantages, strict=False):
                if advantage == 0.0:
                    continue
                batch_datums.append(rollout_to_datum(prompt_tokens, rollout, advantage))

        total_datums += len(batch_datums)
        if not batch_datums and not anchor_datums:
            continue

        for _inner in range(config.inner_epochs):
            if batch_datums:
                fb = training_client.forward_backward(
                    batch_datums,
                    loss_fn=config.loss_fn,
                    loss_fn_config=config.loss_fn_config,
                ).result()
                losses.append(float(fb.loss))
            if anchor_datums:
                anchor_batch = _scaled_anchor_batch(
                    anchor_datums,
                    rng=rng,
                    scale=config.anchor_weight,
                    sample_size=min(len(anchor_datums), max(1, len(batch_datums) or config.prompts_per_batch)),
                )
                if anchor_batch:
                    anchor_fb = training_client.forward_backward(anchor_batch, loss_fn="cross_entropy").result()
                    losses.append(float(anchor_fb.loss))
            training_client.optim_step(AdamParams(learning_rate=config.learning_rate)).result()

    n_scored = len(scored_totals)
    mean_reward = sum(scored_totals) / n_scored if n_scored else 0.0
    per_term_means = {name: total / n_scored if n_scored else 0.0 for name, total in sorted(per_term_totals.items())}
    return GrpoRoundReport(
        mean_reward=mean_reward,
        per_term_means=per_term_means,
        n_datums=total_datums,
        losses=losses,
        n_scored_samples=n_scored,
    )


def _scaled_anchor_batch(
    anchor_datums: list[Datum],
    *,
    rng: random.Random,
    scale: float,
    sample_size: int,
) -> list[Datum]:
    batch = list(anchor_datums) if len(anchor_datums) <= sample_size else rng.sample(anchor_datums, sample_size)
    # Normalize to per-token mean scale BEFORE applying the anchor weight, so the
    # anchor contributes ~anchor_weight x mean-token CE instead of a raw sum that
    # grows with target length and swamps the policy loss (NaNs in bf16).
    batch = _batch_token_normalized(batch)
    scaled: list[Datum] = []
    for datum in batch:
        scaled.append(_scale_datum_weights(datum, scale))
    return scaled


def _scale_datum_weights(datum: Datum, scale: float) -> Datum:
    weights = datum.loss_fn_inputs.get("weights")
    if weights is None:
        source = datum.loss_fn_inputs.get("labels", datum.loss_fn_inputs.get("target_tokens"))
        if source is None:
            return Datum(
                model_input=datum.model_input,
                loss_fn_inputs={**datum.loss_fn_inputs, "weights": [scale]},
                metadata=dict(datum.metadata),
            )
        values = [0.0 if int(value) == -100 else scale for value in _tolist(source)]
    else:
        values = [float(value) * scale for value in _tolist(weights)]
    loss_fn_inputs = dict(datum.loss_fn_inputs)
    loss_fn_inputs["weights"] = values
    return Datum(model_input=datum.model_input, loss_fn_inputs=loss_fn_inputs, metadata=dict(datum.metadata))


def _tolist(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    return list(value)


def _encode_prompt(tokenizer: Any, prompt_text: str) -> list[int]:
    if tokenizer is None:
        return [int(byte) for byte in prompt_text.encode("utf-8", errors="replace")]
    encoded = tokenizer(prompt_text, add_special_tokens=True)
    if isinstance(encoded, dict):
        tokens = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        tokens = encoded.input_ids
    else:
        tokens = encoded
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(token) for token in tokens]


__all__ = [
    "GrpoConfig",
    "GrpoRoundReport",
    "PromptSpec",
    "grpo_round",
]
