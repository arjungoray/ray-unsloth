"""Reusable recipe helpers for supervised and preference optimization."""

from ray_unsloth.recipes.advantages import drop_uniform_groups, group_relative, length_normalized_weights
from ray_unsloth.recipes.datasets import load_dataset, load_examples_block, load_jsonl
from ray_unsloth.recipes.grpo import GrpoConfig, GrpoRoundReport, PromptSpec, grpo_round
from ray_unsloth.recipes.renderers import RENDERER_REGISTRY, Renderer, TrainOnWhat, get_renderer
from ray_unsloth.recipes.rewards import RewardBreakdown, RewardFn, Rubric, RubricTerm
from ray_unsloth.recipes.rollouts import Rollout, collect_group, rollout_to_datum
from ray_unsloth.recipes.sft import conversation_to_datum, sft_epoch, text_completion_datum

__all__ = [
    "RENDERER_REGISTRY",
    "GrpoConfig",
    "GrpoRoundReport",
    "PromptSpec",
    "Renderer",
    "RewardBreakdown",
    "RewardFn",
    "Rollout",
    "Rubric",
    "RubricTerm",
    "TrainOnWhat",
    "collect_group",
    "conversation_to_datum",
    "drop_uniform_groups",
    "get_renderer",
    "group_relative",
    "grpo_round",
    "length_normalized_weights",
    "load_dataset",
    "load_examples_block",
    "load_jsonl",
    "rollout_to_datum",
    "sft_epoch",
    "text_completion_datum",
]
