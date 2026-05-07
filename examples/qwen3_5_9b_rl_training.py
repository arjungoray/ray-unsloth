"""Run a Qwen3.5 9B first-RL training loop on 2 L4 GPUs.

This follows the same primitive flow as the upstream Tinker first-RL tutorial
and ``examples/tinker_first_rl_training.py``:

1. create a LoRA training client
2. sample groups of completions directly from the live policy actor
3. grade completions with a verifiable reward function
4. build RL ``Datum`` records with old logprobs and group-relative advantages
5. submit ``forward_backward_async(..., loss_fn="importance_sampling")`` and
   ``optim_step_async`` before awaiting

Run:

    python examples/qwen3_5_9b_rl_training.py \
        --config configs/qwen3_5_9b_2x_l4_sharded.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import math
import re
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ray_unsloth import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ModelInput,
    SamplingClient,
    SamplingParams,
    ServiceClient,
    TensorData,
)


warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message="Calling super")


BASE_MODEL = "qwen3.5-9b-instruct"
LORA_RANK = 16
NUM_STEPS = 25
BATCH_SIZE = 2
GROUP_SIZE = 4
LEARNING_RATE = 4e-5
MAX_TOKENS = 96
TEMPERATURE = 0.8
TOP_P = 0.95
EXPLORATION_TEMPERATURE = 1.0
EXPLORATION_TOP_P = 0.95
SAMPLER_NAME = "qwen3.5-9b-rl"
MIN_TRAIN_DATUMS = 1
DEGENERATE_REWARD_BASELINE = 0.0
SFT_ANCHOR_WEIGHT = 0.2
WANDB_ENABLED = True
WANDB_PROJECT = "ray-unsloth-rl"
WANDB_RUN_NAME = "qwen3.5-9b-2xl4-rl"
WANDB_LOG_COMPLETIONS = 64


SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step. Put only "
    "the final numerical answer inside \\boxed{} with no units."
)


QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."


FEWSHOT_PREFIX: list[dict[str, str]] = [
    {
        "role": "user",
        "content": "Compute 18 * 7 - 26. Provide a numerical answer without units, written inside \\boxed{}.",
    },
    {
        "role": "assistant",
        "content": "18 * 7 = 126. Then 126 - 26 = 100. The final answer is \\boxed{100}.",
    },
    {
        "role": "user",
        "content": "Solve for x: 3x + 4 = 25. Provide a numerical answer without units, written inside \\boxed{}.",
    },
    {
        "role": "assistant",
        "content": "Subtract 4 from both sides to get 3x = 21, so x = 7. The final answer is \\boxed{7}.",
    },
]


@dataclass(frozen=True)
class MathProblem:
    question: str
    answer: str


@dataclass(frozen=True)
class GradedCompletion:
    tokens: list[int]
    logprobs: list[float]
    text: str
    reward: float
    advantage: float


@dataclass(frozen=True)
class ProblemRollout:
    problem: MathProblem
    prompt: ModelInput
    completions: list[GradedCompletion]
    mean_reward: float
    degenerate: bool


@dataclass
class WandbLogger:
    enabled: bool
    run: Any | None = None
    wandb: Any | None = None
    max_completion_rows: int = WANDB_LOG_COMPLETIONS
    event_index: int = 0

    @classmethod
    def from_settings(
        cls,
        *,
        settings: dict[str, Any],
        config_path: str,
        run_config: dict[str, Any],
    ) -> "WandbLogger":
        wandb_settings = dict(settings.get("wandb", {}))
        enabled = bool(wandb_settings.get("enabled", WANDB_ENABLED))
        if not enabled:
            return cls(enabled=False)
        try:
            import wandb
        except ImportError:
            print("W&B logging requested but wandb is not installed. Install with `pip install wandb`.")
            return cls(enabled=False)

        project = str(wandb_settings.get("project", WANDB_PROJECT))
        run_name = str(wandb_settings.get("name", WANDB_RUN_NAME))
        entity = wandb_settings.get("entity")
        tags = list(wandb_settings.get("tags", ["rl", "qwen3.5", "lora", "2xl4"]))
        notes = wandb_settings.get("notes")
        mode = wandb_settings.get("mode")
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags,
            notes=notes,
            mode=mode,
            config={"config_path": config_path, **run_config, "wandb": wandb_settings},
        )
        return cls(
            enabled=True,
            run=run,
            wandb=wandb,
            max_completion_rows=int(wandb_settings.get("log_completions", WANDB_LOG_COMPLETIONS)),
        )

    def define_metrics(self) -> None:
        if not self.enabled or self.wandb is None:
            return
        self.wandb.define_metric("train/step")
        self.wandb.define_metric("wandb/event_index")
        for prefix in ("reward", "rollout", "policy", "timing", "data", "sampling", "train", "progress", "tokens"):
            self.wandb.define_metric(f"{prefix}/*", step_metric="train/step")

    def log(self, payload: dict[str, Any], *, step: int) -> None:
        if not self.enabled or self.run is None:
            return
        self.event_index += 1
        self.run.log(
            {"wandb/event_index": self.event_index, "train/step": step, **payload},
            step=self.event_index,
        )

    def log_progress(
        self,
        phase: str,
        *,
        step: int,
        step_start: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "progress/phase": phase,
            f"progress/{phase}": 1.0,
            "timing/wall_time": time.time(),
        }
        if step_start is not None:
            payload["timing/step_elapsed_so_far"] = time.time() - step_start
        if extra:
            payload.update(extra)
        self.log(payload, step=step)

    def table(self, columns: list[str], rows: list[list[Any]]):
        if not self.enabled or self.wandb is None:
            return None
        return self.wandb.Table(columns=columns, data=rows[: self.max_completion_rows])

    def histogram(self, values: list[float]):
        if not self.enabled or self.wandb is None or not values:
            return None
        return self.wandb.Histogram(values)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()


PROBLEMS = [
    MathProblem("Compute 37 * 24 - 156.", "732"),
    MathProblem("A sequence starts at 7. Each next term is double the previous term minus 3. What is the 5th term?", "67"),
    MathProblem("Solve for x: 5x - 7 = 3x + 29.", "18"),
    MathProblem("What is the remainder when 17^3 is divided by 19?", "11"),
    MathProblem("A rectangle has perimeter 90. Its length is 3 more than twice its width. What is its area?", "434"),
    MathProblem("The average of 6 numbers is 18. Five numbers are 11, 14, 20, 21, and 25. What is the sixth?", "17"),
    MathProblem("A jar has red and blue marbles in a 3:5 ratio. After adding 8 red marbles, the ratio is 5:7. What was the initial total number of marbles?", "112"),
    MathProblem("Compute 125% of 64, then subtract 17.", "63"),
]


def load_local_settings(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return dict(data.get("examples", {}).get("qwen3_5_9b_rl_training", {}))


def live_training_sampling_client(training_client: Any, *, name: str = "live-policy") -> SamplingClient:
    actor = getattr(training_client, "_actor", None)
    if actor is None:
        raise TypeError("Training client does not expose a live actor for policy sampling")
    return SamplingClient(session_id=f"{training_client.session_id}-{name}", actors=[actor])


def token_ids_from_output(output: Any) -> list[int]:
    if isinstance(output, dict):
        output = output["input_ids"]
    elif hasattr(output, "input_ids"):
        output = output.input_ids
    if hasattr(output, "detach"):
        output = output.detach().cpu()
    if hasattr(output, "tolist"):
        output = output.tolist()
    if isinstance(output, tuple):
        output = list(output)
    if output and isinstance(output[0], list):
        if len(output) != 1:
            raise ValueError(f"Expected one tokenized example, got batch of {len(output)}")
        output = output[0]
    return [int(token) for token in output]


def render_fallback(messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    lines = []
    for message in messages:
        role = message["role"].capitalize()
        lines.append(f"{role}: {message['content']}")
    if add_generation_prompt:
        lines.append("Assistant:")
    return "\n\n".join(lines)


def encode_chat(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            tokens = apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            tokens = apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        return token_ids_from_output(tokens)
    encoded = tokenizer(
        render_fallback(messages, add_generation_prompt=add_generation_prompt),
        add_special_tokens=True,
    )
    return token_ids_from_output(encoded)


def strip_trailing_eos(tokens: list[int], tokenizer: Any) -> list[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and tokens and tokens[-1] == eos_token_id:
        return tokens[:-1]
    return tokens


def decode_tokens(tokenizer: Any, tokens: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        return str(decode(tokens, skip_special_tokens=True))
    batch_decode = getattr(tokenizer, "batch_decode", None)
    if callable(batch_decode):
        decoded = batch_decode([tokens], skip_special_tokens=True)
        return str(decoded[0]) if decoded else ""
    return " ".join(str(token) for token in tokens)


def build_generation_prompt(tokenizer: Any, problem: MathProblem) -> ModelInput:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEWSHOT_PREFIX,
        {"role": "user", "content": problem.question + QUESTION_SUFFIX},
    ]
    return ModelInput.from_ints(
        strip_trailing_eos(
            encode_chat(tokenizer, messages, add_generation_prompt=True),
            tokenizer,
        )
    )


def extract_boxed(text: str) -> str | None:
    match = re.findall(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match[-1].strip()
    return None


def extract_numeric_answer(text: str) -> tuple[str | None, bool]:
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed, True

    answer_patterns = [
        r"(?:final\s+answer|answer|therefore|so)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*(?:is\s+the\s+(?:final\s+)?answer)",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return str(matches[-1]).strip(), False

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1], False
    return None, False


def normalize_numeric(text: str) -> str:
    return text.replace(",", "").strip()


def numeric_value(text: str) -> float | None:
    cleaned = normalize_numeric(text)
    try:
        return float(cleaned)
    except ValueError:
        return None


def grade_answer(response: str, ground_truth: str) -> float:
    answer, used_box = extract_numeric_answer(response)
    if answer is None:
        return -0.5
    if normalize_numeric(answer) == normalize_numeric(ground_truth):
        word_count = len(response.split())
        brevity_bonus = max(0.0, 0.2 - min(word_count, 120) / 600.0)
        format_bonus = 0.4 if used_box else 0.0
        return 0.8 + format_bonus + brevity_bonus
    predicted = numeric_value(answer)
    expected = numeric_value(ground_truth)
    if predicted is None or expected is None:
        return -0.45
    error = abs(predicted - expected)
    closeness = max(0.0, 1.0 - error / max(1.0, abs(expected)))
    format_bonus = 0.1 if used_box else 0.0
    return -0.35 + format_bonus + 0.7 * closeness


def group_relative_advantages(
    rewards: Sequence[float],
    *,
    degenerate_baseline: float | None = None,
) -> list[float]:
    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    advantages = [float(reward) - mean_reward for reward in rewards]
    if any(advantage != 0.0 for advantage in advantages) or degenerate_baseline is None:
        return advantages
    return [float(reward) - degenerate_baseline for reward in rewards]


def finite_logprobs(logprobs: Sequence[float | None] | None, target_length: int) -> list[float]:
    values = [0.0 if value is None else float(value) for value in (logprobs or [])]
    if len(values) < target_length:
        values.extend([0.0] * (target_length - len(values)))
    return values[:target_length]


def build_policy_datum(
    *,
    prompt: ModelInput,
    completion_tokens: Sequence[int],
    completion_logprobs: Sequence[float | None] | None,
    advantage: float,
) -> Datum:
    if not completion_tokens:
        raise ValueError("completion_tokens must not be empty")

    tokens = [int(token) for token in completion_tokens]
    prompt_target_padding = max(prompt.length - 1, 0)
    model_input = prompt.append(EncodedTextChunk(tokens=tokens[:-1]))
    target_tokens = [0] * prompt_target_padding + tokens
    old_logprobs = [0.0] * prompt_target_padding + finite_logprobs(completion_logprobs, len(tokens))
    advantages = [0.0] * prompt_target_padding + [float(advantage)] * len(tokens)
    weights = [0.0] * prompt_target_padding + [1.0] * len(tokens)

    return Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData(data=target_tokens, dtype="int64", shape=[len(target_tokens)]),
            "logprobs": TensorData(data=old_logprobs, dtype="float32", shape=[len(old_logprobs)]),
            "advantages": TensorData(data=advantages, dtype="float32", shape=[len(advantages)]),
            "weights": TensorData(data=weights, dtype="float32", shape=[len(weights)]),
        },
    )


def encode_plain_text(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        return token_ids_from_output(encode(text, add_special_tokens=add_special_tokens))
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    return token_ids_from_output(encoded)


def canonical_solution(problem: MathProblem) -> str:
    return f"The final answer is \\boxed{{{problem.answer}}}."


def build_supervised_datum(
    *,
    tokenizer: Any,
    problem: MathProblem,
    weight: float = SFT_ANCHOR_WEIGHT,
) -> Datum:
    prompt = build_generation_prompt(tokenizer, problem)
    completion_tokens = encode_plain_text(tokenizer, canonical_solution(problem), add_special_tokens=False)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and (not completion_tokens or completion_tokens[-1] != eos_token_id):
        completion_tokens.append(int(eos_token_id))
    if not completion_tokens:
        raise ValueError("Supervised completion must not be empty")

    prompt_target_padding = max(prompt.length - 1, 0)
    model_input = prompt.append(EncodedTextChunk(tokens=completion_tokens[:-1]))
    target_tokens = [0] * prompt_target_padding + completion_tokens
    weights = [0.0] * prompt_target_padding + [float(weight)] * len(completion_tokens)
    return Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData(data=target_tokens, dtype="int64", shape=[len(target_tokens)]),
            "weights": TensorData(data=weights, dtype="float32", shape=[len(weights)]),
        },
    )


def rollout_to_datums(rollout: ProblemRollout) -> list[Datum]:
    return [
        build_policy_datum(
            prompt=rollout.prompt,
            completion_tokens=completion.tokens,
            completion_logprobs=completion.logprobs,
            advantage=completion.advantage,
        )
        for completion in rollout.completions
        if completion.tokens and completion.advantage != 0.0
    ]


def policy_loss_summary(loss_fn_outputs: list[dict[str, TensorData]]) -> tuple[float, float]:
    logprob_count = 0
    logprob_sum = 0.0
    ratio_count = 0
    ratio_sum = 0.0
    for output in loss_fn_outputs:
        for value in output["logprobs"].tolist():
            if float(value) != 0.0:
                logprob_sum += float(value)
                logprob_count += 1
        ratios = output.get("ratios")
        if ratios is not None:
            for value in ratios.tolist():
                if float(value) != 0.0:
                    ratio_sum += float(value)
                    ratio_count += 1
    mean_logprob = logprob_sum / logprob_count if logprob_count else 0.0
    mean_ratio = ratio_sum / ratio_count if ratio_count else 0.0
    return mean_logprob, mean_ratio


def rollout_flat_rows(rollouts: Sequence[ProblemRollout]) -> list[dict[str, Any]]:
    rows = []
    for problem_index, rollout in enumerate(rollouts):
        for completion_index, completion in enumerate(rollout.completions):
            rows.append(
                {
                    "problem_index": problem_index,
                    "completion_index": completion_index,
                    "question": rollout.problem.question,
                    "answer": rollout.problem.answer,
                    "text": completion.text,
                    "reward": completion.reward,
                    "advantage": completion.advantage,
                    "tokens": len(completion.tokens),
                    "mean_reward": rollout.mean_reward,
                    "degenerate": rollout.degenerate,
                }
            )
    return rows


def scalar_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": float(mean),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(math.sqrt(variance)),
    }


def wandb_step_payload(
    *,
    step: int,
    rollouts: Sequence[ProblemRollout],
    datums: Sequence[Datum],
    expert_datums: Sequence[Datum],
    used_exploration_retry: bool,
    mean_logprob: float,
    mean_ratio: float,
    training_loss: float,
    rl_loss: float,
    sft_loss: float,
    optimizer_step: int | None,
    elapsed: float,
    learning_rate: float,
    sampling_params: SamplingParams,
    logger: WandbLogger,
) -> dict[str, Any]:
    rows = rollout_flat_rows(rollouts)
    rewards = [float(row["reward"]) for row in rows]
    advantages = [float(row["advantage"]) for row in rows]
    tokens = [float(row["tokens"]) for row in rows]
    reward_summary = scalar_summary(rewards)
    advantage_summary = scalar_summary(advantages)
    token_summary = scalar_summary(tokens)
    completion_table = logger.table(
        columns=[
            "step",
            "problem_index",
            "completion_index",
            "question",
            "answer",
            "reward",
            "advantage",
            "tokens",
            "degenerate",
            "text",
        ],
        rows=[
            [
                step,
                row["problem_index"],
                row["completion_index"],
                row["question"],
                row["answer"],
                row["reward"],
                row["advantage"],
                row["tokens"],
                row["degenerate"],
                row["text"],
            ]
            for row in rows
        ],
    )
    payload: dict[str, Any] = {
        "reward/mean": reward_summary["mean"],
        "reward/min": reward_summary["min"],
        "reward/max": reward_summary["max"],
        "reward/std": reward_summary["std"],
        "reward/nonzero_fraction": sum(1 for reward in rewards if reward != 0.0) / len(rewards) if rewards else 0.0,
        "reward/histogram": logger.histogram(rewards),
        "rollout/problem_count": len(rollouts),
        "rollout/completion_count": len(rows),
        "rollout/degenerate_count": sum(1 for rollout in rollouts if rollout.degenerate),
        "rollout/degenerate_fraction": sum(1 for rollout in rollouts if rollout.degenerate) / len(rollouts)
        if rollouts
        else 0.0,
        "rollout/used_exploration_retry": 1.0 if used_exploration_retry else 0.0,
        "rollout/completion_tokens_mean": token_summary["mean"],
        "rollout/completion_tokens_max": token_summary["max"],
        "rollout/advantage_mean": advantage_summary["mean"],
        "rollout/advantage_min": advantage_summary["min"],
        "rollout/advantage_max": advantage_summary["max"],
        "rollout/advantage_std": advantage_summary["std"],
        "rollout/advantage_histogram": logger.histogram(advantages),
        "policy/mean_logprob": mean_logprob,
        "policy/mean_ratio": mean_ratio,
        "data/datums": len(datums),
        "data/rl_datums": len(datums),
        "data/expert_datums": len(expert_datums),
        "data/nonempty_datums": sum(1 for datum in datums if datum.model_input.length > 0),
        "sampling/temperature": sampling_params.temperature,
        "sampling/top_p": sampling_params.top_p,
        "sampling/max_tokens": sampling_params.max_tokens,
        "train/loss": training_loss,
        "train/rl_loss": rl_loss,
        "train/sft_anchor_loss": sft_loss,
        "train/learning_rate": learning_rate,
        "train/updated": 1.0 if optimizer_step is not None else 0.0,
        "train/optimizer_step": optimizer_step,
        "timing/step_seconds": elapsed,
    }
    if completion_table is not None:
        payload["rollout/completions"] = completion_table
    return {key: value for key, value in payload.items() if value is not None}


async def collect_rollouts(
    *,
    tokenizer: Any,
    sampling_client: Any,
    problems: Sequence[MathProblem],
    group_size: int,
    sampling_params: SamplingParams,
    degenerate_reward_baseline: float | None = DEGENERATE_REWARD_BASELINE,
) -> list[ProblemRollout]:
    sample_coros = []
    prompts = []
    for problem in problems:
        prompt = build_generation_prompt(tokenizer, problem)
        prompts.append(prompt)
        sample_coros.append(
            sampling_client.sample_async(
                prompt=prompt,
                num_samples=group_size,
                sampling_params=sampling_params,
            )
        )

    sample_results = await asyncio.gather(*sample_coros)
    rollouts = []
    for problem, prompt, sample_result in zip(problems, prompts, sample_results):
        rewards = []
        raw_sequences = []
        for sequence in sample_result.sequences:
            text = sequence.text or decode_tokens(tokenizer, sequence.tokens)
            reward = grade_answer(text, problem.answer)
            rewards.append(reward)
            raw_sequences.append((sequence, text, reward))

        group_advantages = group_relative_advantages(rewards)
        degenerate = all(advantage == 0.0 for advantage in group_advantages)
        advantages = (
            group_relative_advantages(rewards, degenerate_baseline=degenerate_reward_baseline)
            if degenerate
            else group_advantages
        )
        completions = [
            GradedCompletion(
                tokens=list(sequence.tokens),
                logprobs=finite_logprobs(sequence.logprobs, len(sequence.tokens)),
                text=text,
                reward=reward,
                advantage=advantage,
            )
            for (sequence, text, reward), advantage in zip(raw_sequences, advantages)
        ]
        rollouts.append(
            ProblemRollout(
                problem=problem,
                prompt=prompt,
                completions=completions,
                mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
                degenerate=degenerate,
            )
        )
    return rollouts


async def train(args: argparse.Namespace) -> None:
    settings = load_local_settings(args.config)
    num_steps = int(settings.get("steps", NUM_STEPS))
    batch_size = int(settings.get("batch_size", BATCH_SIZE))
    group_size = int(settings.get("group_size", GROUP_SIZE))
    learning_rate = float(settings.get("learning_rate", LEARNING_RATE))
    sampler_name = str(settings.get("name", SAMPLER_NAME))
    max_tokens = int(settings.get("max_tokens", MAX_TOKENS))
    temperature = float(settings.get("temperature", TEMPERATURE))
    top_p = float(settings.get("top_p", TOP_P))
    exploration_temperature = float(settings.get("exploration_temperature", EXPLORATION_TEMPERATURE))
    exploration_top_p = float(settings.get("exploration_top_p", EXPLORATION_TOP_P))
    min_train_datums = int(settings.get("min_train_datums", MIN_TRAIN_DATUMS))
    sft_anchor_weight = float(settings.get("sft_anchor_weight", SFT_ANCHOR_WEIGHT))
    wandb_logger = WandbLogger.from_settings(
        settings=settings,
        config_path=str(args.config),
        run_config={
            "base_model": BASE_MODEL,
            "lora_rank": LORA_RANK,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "group_size": group_size,
            "learning_rate": learning_rate,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "exploration_temperature": exploration_temperature,
            "exploration_top_p": exploration_top_p,
            "min_train_datums": min_train_datums,
            "degenerate_reward_baseline": DEGENERATE_REWARD_BASELINE,
            "sft_anchor_weight": sft_anchor_weight,
            "problem_count": len(PROBLEMS),
        },
    )
    wandb_logger.define_metrics()
    wandb_logger.log_progress(
        "run_started",
        step=0,
        extra={
            "sampling/batch_size": batch_size,
            "sampling/group_size": group_size,
            "sampling/max_tokens": max_tokens,
        },
    )

    service_client = ServiceClient(config=args.config)
    try:
        wandb_logger.log_progress("model_create_started", step=0)
        training_client = await service_client.create_lora_training_client_async(
            base_model=BASE_MODEL,
            rank=LORA_RANK,
        )
        tokenizer = training_client.get_tokenizer()
        sampling_client = live_training_sampling_client(training_client, name=sampler_name)
        wandb_logger.log_progress("model_ready", step=0)
        adam_params = AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95)
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        exploration_sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=max(temperature, exploration_temperature),
            top_p=exploration_top_p,
        )

        for step in range(num_steps):
            t0 = time.time()
            start = (step * batch_size) % len(PROBLEMS)
            batch = [PROBLEMS[(start + offset) % len(PROBLEMS)] for offset in range(batch_size)]
            wandb_logger.log_progress(
                "step_started",
                step=step,
                step_start=t0,
                extra={"data/problem_count": len(batch)},
            )
            wandb_logger.log_progress("sampling_started", step=step, step_start=t0)

            rollouts = await collect_rollouts(
                tokenizer=tokenizer,
                sampling_client=sampling_client,
                problems=batch,
                group_size=group_size,
                sampling_params=sampling_params,
                degenerate_reward_baseline=DEGENERATE_REWARD_BASELINE,
            )
            datums = [datum for rollout in rollouts for datum in rollout_to_datums(rollout)]
            expert_datums = [
                build_supervised_datum(tokenizer=tokenizer, problem=problem, weight=sft_anchor_weight)
                for problem in batch
                if sft_anchor_weight > 0.0
            ]
            used_exploration_retry = False
            initial_rows = rollout_flat_rows(rollouts)
            wandb_logger.log_progress(
                "rollouts_collected",
                step=step,
                step_start=t0,
                extra={
                    "reward/pre_retry_mean": scalar_summary([float(row["reward"]) for row in initial_rows])["mean"],
                    "rollout/pre_retry_degenerate_fraction": (
                        sum(1 for rollout in rollouts if rollout.degenerate) / len(rollouts) if rollouts else 0.0
                    ),
                    "data/pre_retry_datums": len(datums),
                    "data/expert_datums": len(expert_datums),
                },
            )

            if len(datums) < min_train_datums:
                wandb_logger.log_progress(
                    "exploration_retry_started",
                    step=step,
                    step_start=t0,
                    extra={"data/pre_retry_datums": len(datums)},
                )
                rollouts = await collect_rollouts(
                    tokenizer=tokenizer,
                    sampling_client=sampling_client,
                    problems=batch,
                    group_size=group_size,
                    sampling_params=exploration_sampling_params,
                    degenerate_reward_baseline=DEGENERATE_REWARD_BASELINE,
                )
                datums = [datum for rollout in rollouts for datum in rollout_to_datums(rollout)]
                used_exploration_retry = True

            if len(datums) >= min_train_datums or expert_datums:
                wandb_logger.log_progress(
                    "backward_started",
                    step=step,
                    step_start=t0,
                    extra={"data/datums": len(datums), "data/expert_datums": len(expert_datums)},
                )
                rl_future = None
                if datums:
                    rl_future = await training_client.forward_backward_async(
                        datums,
                        loss_fn="importance_sampling",
                    )
                sft_future = None
                if expert_datums:
                    sft_future = await training_client.forward_backward_async(
                        expert_datums,
                        loss_fn="cross_entropy",
                    )
                optim_future = await training_client.optim_step_async(adam_params)
                rl_result = await rl_future.result_async() if rl_future is not None else None
                sft_result = await sft_future.result_async() if sft_future is not None else None
                optim_result = await optim_future.result_async()
                if rl_result is not None:
                    mean_logprob, mean_ratio = policy_loss_summary(rl_result.loss_fn_outputs)
                    rl_loss = rl_result.loss
                else:
                    mean_logprob = 0.0
                    mean_ratio = 0.0
                    rl_loss = 0.0
                sft_loss = sft_result.loss if sft_result is not None else 0.0
                training_loss = rl_loss + sft_loss
                optimizer_step = optim_result.step
                wandb_logger.log_progress(
                    "optimizer_finished",
                    step=step,
                    step_start=t0,
                    extra={
                        "policy/mean_logprob": mean_logprob,
                        "policy/mean_ratio": mean_ratio,
                        "train/rl_loss": rl_loss,
                        "train/sft_anchor_loss": sft_loss,
                    },
                )
            else:
                mean_logprob = 0.0
                mean_ratio = 0.0
                rl_loss = 0.0
                sft_loss = 0.0
                training_loss = 0.0
                optimizer_step = None
                wandb_logger.log_progress(
                    "train_skipped_no_datums",
                    step=step,
                    step_start=t0,
                    extra={"data/datums": len(datums)},
                )

            mean_reward = sum(rollout.mean_reward for rollout in rollouts) / len(rollouts)
            frac_degenerate = sum(1 for rollout in rollouts if rollout.degenerate) / len(rollouts)
            elapsed = time.time() - t0
            wandb_logger.log(
                wandb_step_payload(
                    step=step,
                    rollouts=rollouts,
                    datums=datums,
                    expert_datums=expert_datums,
                    used_exploration_retry=used_exploration_retry,
                    mean_logprob=mean_logprob,
                    mean_ratio=mean_ratio,
                    training_loss=training_loss,
                    rl_loss=rl_loss,
                    sft_loss=sft_loss,
                    optimizer_step=optimizer_step,
                    elapsed=elapsed,
                    learning_rate=learning_rate,
                    sampling_params=exploration_sampling_params if used_exploration_retry else sampling_params,
                    logger=wandb_logger,
                ),
                step=step,
            )
            print(
                f"Step {step:2d}: reward={mean_reward:.3f} "
                f"degenerate={frac_degenerate:.0%} datums={len(datums)} expert={len(expert_datums)} "
                f"train_min={min_train_datums} retry={used_exploration_retry} "
                f"mean_logprob={mean_logprob:.3f} mean_ratio={mean_ratio:.3f} "
                f"({elapsed:.1f}s)"
            )
    finally:
        wandb_logger.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_5_9b_2x_l4_sharded.yaml")
    args = parser.parse_args()

    asyncio.run(train(args))


if __name__ == "__main__":
    main()
