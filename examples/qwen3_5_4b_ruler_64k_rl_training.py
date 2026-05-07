"""Run Qwen3.5 4B RL on RULER 64K long-context retrieval.

This example uses ``tonychenxyz/ruler-full``. The dataset already contains
Qwen3-4B chat-formatted prompts and exact ground-truth answers, which makes it a
good fit for RL: reward is a deterministic string-match score instead of a fuzzy
judge. The default config filters for 64K-context samples whose tokenized input
is at least 40K tokens.

Run:

    python examples/qwen3_5_4b_ruler_64k_rl_training.py \
        --config configs/qwen3_5_4b_ruler_64k.yaml \
        --dataset-limit 16
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ray_unsloth import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ForwardBackwardOutput,
    ModelInput,
    SamplingParams,
    ServiceClient,
    TensorData,
)


_HELPER_PATH = Path(__file__).with_name("qwen3_5_9b_rl_training.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("qwen3_5_9b_rl_training_helpers", _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError(f"Could not load helper example from {_HELPER_PATH}")
_helpers = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _helpers
_HELPER_SPEC.loader.exec_module(_helpers)


BASE_MODEL = "qwen3.5-4b-ruler-64k"
LORA_RANK = 8
DATASET_NAME = "tonychenxyz/ruler-full"
DATASET_CONFIG = "plain"
DATASET_SPLIT = "validation"
RULER_TASK = "niah_single_1"
CONTEXT_LENGTH = 65536
DATASET_LIMIT = 16
DATASET_SCAN_LIMIT = 40000
MIN_PROMPT_TOKENS = 40000
MAX_PROMPT_TOKENS: int | None = None
NUM_STEPS = 20
BATCH_SIZE = 1
GROUP_SIZE = 2
LEARNING_RATE = 2e-5
MAX_TOKENS = 4096
MAX_TRAIN_TOKENS = 73728
TRAIN_MICROBATCH_SIZE = 1
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
MAX_TIME = 300.0
MIN_TRAIN_DATUMS = 1
DEGENERATE_REWARD_BASELINE = 0.0
SFT_ANCHOR_WEIGHT = 0.0
STOP_SEQUENCES = ["<|im_end|>"]
SAMPLER_NAME = "qwen3.5-4b-ruler-64k-rl"
WANDB_PROJECT = "ray-unsloth-rl"
WANDB_RUN_NAME = "qwen3.5-4b-ruler-64k-rl"
WANDB_LOG_COMPLETIONS = 64


WandbLogger = _helpers.WandbLogger


@dataclass(slots=True)
class RulerProblem:
    prompt: ModelInput
    prompt_text: str
    answers: list[str]
    task: str
    context_length: int | None
    category: str
    prompt_tokens: int
    row_index: int


@dataclass(slots=True)
class RulerCompletion:
    tokens: list[int]
    logprobs: list[float | None]
    text: str
    reward: float
    advantage: float


@dataclass(slots=True)
class RulerRollout:
    problem: RulerProblem
    completions: list[RulerCompletion]
    mean_reward: float
    degenerate: bool


@dataclass(slots=True)
class DatasetInfo:
    name: str
    config: str
    split: str
    selected_rows: int
    scanned_rows: int
    task: str
    context_length: int
    min_prompt_tokens: int


def load_local_settings(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return dict(data.get("examples", {}).get("qwen3_5_4b_ruler_64k_rl_training", {}))


def _setting(
    args: argparse.Namespace,
    settings: dict[str, Any],
    key: str,
    default: Any,
    cast_type: type,
) -> Any:
    value = getattr(args, key, None)
    if value is None:
        value = settings.get(key, default)
    if value is None:
        return None
    return cast_type(value)


def _parse_extra_info(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _context_length_from_category(category: str) -> int | None:
    match = re.search(r"(?:^|_)(\d{4,6})$", category)
    return int(match.group(1)) if match else None


def _task_from_extra(extra: dict[str, Any], category: str) -> str:
    value = extra.get("task") or extra.get("ruler_task") or extra.get("name")
    if isinstance(value, str) and value:
        return value
    match = re.search(r"ruler/([^/]+)_\d{4,6}$", category)
    if match:
        return match.group(1)
    parts = category.rsplit("/", maxsplit=1)
    tail = parts[-1] if parts else category
    return re.sub(r"_\d{4,6}$", "", tail)


def _answers_from_extra(extra: dict[str, Any]) -> list[str]:
    ground_truth = extra.get("ground_truth")
    if isinstance(ground_truth, dict):
        answers = ground_truth.get("answers") or ground_truth.get("answer")
    else:
        answers = ground_truth
    if answers is None:
        answers = extra.get("answers") or extra.get("answer")
    if isinstance(answers, str):
        return [answers]
    if isinstance(answers, Sequence):
        return [str(answer) for answer in answers if str(answer)]
    return []


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def score_ruler_response(text: str, answers: Sequence[str]) -> float:
    if not answers:
        return 0.0
    normalized_text = _normalize_for_match(text)
    normalized_answers = [_normalize_for_match(answer) for answer in answers if str(answer).strip()]
    if not normalized_answers:
        return 0.0
    return 1.0 if all(answer in normalized_text for answer in normalized_answers) else 0.0


def load_ruler_problems(
    *,
    tokenizer: Any,
    dataset_name: str,
    dataset_config: str,
    split: str,
    task: str,
    context_length: int,
    limit: int,
    scan_limit: int,
    min_prompt_tokens: int,
    max_prompt_tokens: int | None,
) -> tuple[list[RulerProblem], DatasetInfo]:
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - exercised without examples deps
        raise RuntimeError("Install example dependencies with `pip install -e '.[examples]'`.") from exc

    rows = datasets.load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    problems: list[RulerProblem] = []
    scanned_rows = 0
    seen_categories: dict[str, int] = {}
    matching_task_context_rows = 0
    below_min_prompt_tokens = 0
    above_max_prompt_tokens = 0
    for row in rows:
        scanned_rows += 1
        extra = _parse_extra_info(row.get("extra_info"))
        category = str(row.get("category") or extra.get("category") or "")
        if len(seen_categories) < 24:
            seen_categories[category] = seen_categories.get(category, 0) + 1
        row_task = _task_from_extra(extra, category)
        row_context_length = extra.get("context_length") or _context_length_from_category(category)
        try:
            row_context_length = int(row_context_length) if row_context_length is not None else None
        except (TypeError, ValueError):
            row_context_length = None
        if row_task != task or row_context_length != context_length:
            if scanned_rows >= scan_limit:
                break
            continue
        matching_task_context_rows += 1
        answers = _answers_from_extra(extra)
        prompt_text = str(row.get("prompt") or row.get("input") or "")
        if not answers or not prompt_text:
            if scanned_rows >= scan_limit:
                break
            continue
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        token_count = len(prompt_tokens)
        if token_count < min_prompt_tokens:
            below_min_prompt_tokens += 1
            if scanned_rows >= scan_limit:
                break
            continue
        if max_prompt_tokens is not None and token_count > max_prompt_tokens:
            above_max_prompt_tokens += 1
            if scanned_rows >= scan_limit:
                break
            continue
        problems.append(
            RulerProblem(
                prompt=ModelInput.from_ints(prompt_tokens),
                prompt_text=prompt_text,
                answers=answers,
                task=row_task,
                context_length=row_context_length,
                category=category,
                prompt_tokens=token_count,
                row_index=scanned_rows - 1,
            )
        )
        if len(problems) >= limit:
            break
        if scanned_rows >= scan_limit:
            break

    if not problems:
        seen = ", ".join(f"{category} ({count})" for category, count in seen_categories.items())
        raise ValueError(
            f"No RULER rows matched task={task!r}, context_length={context_length}, "
            f"min_prompt_tokens={min_prompt_tokens}, max_prompt_tokens={max_prompt_tokens} "
            f"within {scanned_rows} scanned rows. "
            f"matching_task_context_rows={matching_task_context_rows}, "
            f"below_min_prompt_tokens={below_min_prompt_tokens}, "
            f"above_max_prompt_tokens={above_max_prompt_tokens}, "
            f"seen_categories=[{seen}]"
        )
    return problems, DatasetInfo(
        name=dataset_name,
        config=dataset_config,
        split=split,
        selected_rows=len(problems),
        scanned_rows=scanned_rows,
        task=task,
        context_length=context_length,
        min_prompt_tokens=min_prompt_tokens,
    )


def _datum_batches(datums: Sequence[Datum], batch_size: int) -> Iterable[list[Datum]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(datums), batch_size):
        yield list(datums[start : start + batch_size])


async def forward_backward_microbatches(
    training_client: Any,
    datums: Sequence[Datum],
    *,
    loss_fn: str,
    microbatch_size: int,
) -> list[ForwardBackwardOutput]:
    results: list[ForwardBackwardOutput] = []
    for microbatch in _datum_batches(datums, microbatch_size):
        future = await training_client.forward_backward_async(
            microbatch,
            loss_fn=loss_fn,
        )
        results.append(await future.result_async())
    return results


def count_train_tokens(datums: Sequence[Datum]) -> int:
    return sum(int(datum.model_input.length) for datum in datums)


def rollout_to_datums(rollout: RulerRollout, *, max_train_tokens: int) -> list[Datum]:
    datums = []
    for completion in rollout.completions:
        if not completion.tokens or completion.advantage == 0.0:
            continue
        available_completion_tokens = max_train_tokens - max(rollout.problem.prompt.length - 1, 0)
        if available_completion_tokens <= 0:
            continue
        datums.append(
            _helpers.build_policy_datum(
                prompt=rollout.problem.prompt,
                completion_tokens=completion.tokens[:available_completion_tokens],
                completion_logprobs=completion.logprobs[:available_completion_tokens],
                advantage=completion.advantage,
            )
        )
    return datums


def build_supervised_anchor_datum(
    *,
    tokenizer: Any,
    problem: RulerProblem,
    weight: float,
    max_train_tokens: int,
) -> Datum | None:
    if weight <= 0.0 or not problem.answers:
        return None
    answer_text = str(problem.answers[0])
    completion_tokens = _helpers.encode_plain_text(tokenizer, answer_text, add_special_tokens=False)
    if not completion_tokens:
        return None
    available_completion_tokens = max_train_tokens - max(problem.prompt.length - 1, 0)
    if available_completion_tokens <= 0:
        return None
    completion_tokens = completion_tokens[:available_completion_tokens]
    model_input = problem.prompt.append(EncodedTextChunk(tokens=completion_tokens[:-1]))
    prompt_padding = max(problem.prompt.length - 1, 0)
    target_tokens = [0] * prompt_padding + completion_tokens
    weights = [0.0] * prompt_padding + [float(weight)] * len(completion_tokens)
    return Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData(data=target_tokens, dtype="int64", shape=[len(target_tokens)]),
            "weights": TensorData(data=weights, dtype="float32", shape=[len(weights)]),
        },
    )


async def collect_rollouts(
    *,
    tokenizer: Any,
    sampling_client: Any,
    problems: Sequence[RulerProblem],
    group_size: int,
    sampling_params: SamplingParams,
    degenerate_reward_baseline: float | None,
    logger: WandbLogger | None,
    step: int,
    step_start: float,
    token_totals: dict[str, int],
) -> list[RulerRollout]:
    rollouts = []
    for problem in problems:
        rewards = []
        completions = []
        for sample_index in range(group_size):
            if logger is not None:
                logger.log_progress(
                    "sample_started",
                    step=step,
                    step_start=step_start,
                    extra={
                        "rollout/sample_index": sample_index,
                        "tokens/input_per_request": problem.prompt_tokens,
                    },
                )
            sample_result = await sampling_client.sample_async(
                prompt=problem.prompt,
                num_samples=1,
                sampling_params=sampling_params,
            )
            sequence = sample_result.sequences[0]
            text = sequence.text or _helpers.decode_tokens(tokenizer, sequence.tokens)
            reward = score_ruler_response(text, problem.answers)
            token_totals["prefill"] += problem.prompt_tokens
            token_totals["sample"] += len(sequence.tokens)
            rewards.append(reward)
            completions.append((sequence, text, reward))
            if logger is not None:
                logger.log_progress(
                    "sample_finished",
                    step=step,
                    step_start=step_start,
                    extra={
                        "rollout/sample_index": sample_index,
                        "rollout/sample_reward": reward,
                        "rollout/sample_tokens": len(sequence.tokens),
                        "tokens/input_per_request": problem.prompt_tokens,
                        "tokens/prefill_total": token_totals["prefill"],
                        "tokens/sample_total": token_totals["sample"],
                        "tokens/train_total": token_totals["train"],
                    },
                )

        advantages = _helpers.group_relative_advantages(rewards)
        degenerate = all(advantage == 0.0 for advantage in advantages)
        if degenerate:
            advantages = _helpers.group_relative_advantages(
                rewards,
                degenerate_baseline=degenerate_reward_baseline,
            )
        graded = [
            RulerCompletion(
                tokens=list(sequence.tokens),
                logprobs=list(sequence.logprobs or []),
                text=text,
                reward=reward,
                advantage=advantage,
            )
            for (sequence, text, reward), advantage in zip(completions, advantages)
        ]
        rollouts.append(
            RulerRollout(
                problem=problem,
                completions=graded,
                mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
                degenerate=degenerate,
            )
        )
    return rollouts


def wandb_step_payload(
    *,
    step: int,
    rollouts: Sequence[RulerRollout],
    datums: Sequence[Datum],
    expert_datums: Sequence[Datum],
    mean_logprob: float,
    mean_ratio: float,
    training_loss: float,
    rl_loss: float,
    sft_loss: float,
    optimizer_step: int | None,
    elapsed: float,
    learning_rate: float,
    sampling_params: SamplingParams,
    token_totals: dict[str, int],
    logger: WandbLogger,
) -> dict[str, Any]:
    completions = [completion for rollout in rollouts for completion in rollout.completions]
    prompt_tokens = [rollout.problem.prompt_tokens for rollout in rollouts]
    rewards = [completion.reward for completion in completions]
    rows = [
        [
            rollout.problem.row_index,
            rollout.problem.task,
            rollout.problem.prompt_tokens,
            rollout.problem.answers,
            completion.reward,
            completion.advantage,
            len(completion.tokens),
            completion.text,
        ]
        for rollout in rollouts
        for completion in rollout.completions
    ]
    payload: dict[str, Any] = {
        "data/problem_count": len(rollouts),
        "data/datums": len(datums),
        "data/expert_datums": len(expert_datums),
        "reward/mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "reward/max": max(rewards) if rewards else 0.0,
        "rollout/degenerate_fraction": sum(1 for rollout in rollouts if rollout.degenerate) / len(rollouts),
        "rollout/completion_tokens_mean": (
            sum(len(completion.tokens) for completion in completions) / len(completions) if completions else 0.0
        ),
        "tokens/input_per_request_min": min(prompt_tokens) if prompt_tokens else 0,
        "tokens/input_per_request_mean": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0.0,
        "tokens/input_per_request_max": max(prompt_tokens) if prompt_tokens else 0,
        "tokens/prefill_total": token_totals["prefill"],
        "tokens/sample_total": token_totals["sample"],
        "tokens/train_total": token_totals["train"],
        "sampling/max_tokens": sampling_params.max_tokens,
        "sampling/logprobs_max_tokens": sampling_params.logprobs_max_tokens,
        "sampling/top_k": sampling_params.top_k,
        "sampling/max_time": sampling_params.max_time,
        "policy/mean_logprob": mean_logprob,
        "policy/mean_ratio": mean_ratio,
        "train/loss": training_loss,
        "train/rl_loss": rl_loss,
        "train/sft_anchor_loss": sft_loss,
        "train/optimizer_step": optimizer_step,
        "train/learning_rate": learning_rate,
        "timing/step_seconds": elapsed,
    }
    table = logger.table(
        ["row_index", "task", "input_tokens", "answers", "reward", "advantage", "sample_tokens", "completion"],
        rows,
    )
    if table is not None:
        payload["samples/completions"] = table
    histogram = logger.histogram(prompt_tokens)
    if histogram is not None:
        payload["tokens/input_per_request_histogram"] = histogram
    return payload


async def train(args: argparse.Namespace) -> None:
    settings = load_local_settings(args.config)
    dataset_name = _setting(args, settings, "dataset_name", DATASET_NAME, str)
    dataset_config = _setting(args, settings, "dataset_config", DATASET_CONFIG, str)
    dataset_split = _setting(args, settings, "split", DATASET_SPLIT, str)
    ruler_task = _setting(args, settings, "task", RULER_TASK, str)
    context_length = _setting(args, settings, "context_length", CONTEXT_LENGTH, int)
    dataset_limit = _setting(args, settings, "dataset_limit", DATASET_LIMIT, int)
    dataset_scan_limit = _setting(args, settings, "dataset_scan_limit", DATASET_SCAN_LIMIT, int)
    min_prompt_tokens = _setting(args, settings, "min_prompt_tokens", MIN_PROMPT_TOKENS, int)
    max_prompt_tokens = _setting(args, settings, "max_prompt_tokens", MAX_PROMPT_TOKENS, int)
    num_steps = _setting(args, settings, "steps", NUM_STEPS, int)
    batch_size = _setting(args, settings, "batch_size", BATCH_SIZE, int)
    group_size = _setting(args, settings, "group_size", GROUP_SIZE, int)
    learning_rate = _setting(args, settings, "learning_rate", LEARNING_RATE, float)
    max_tokens = _setting(args, settings, "max_tokens", MAX_TOKENS, int)
    max_train_tokens = _setting(args, settings, "max_train_tokens", MAX_TRAIN_TOKENS, int)
    train_microbatch_size = _setting(args, settings, "train_microbatch_size", TRAIN_MICROBATCH_SIZE, int)
    temperature = _setting(args, settings, "temperature", TEMPERATURE, float)
    top_p = _setting(args, settings, "top_p", TOP_P, float)
    top_k = _setting(args, settings, "top_k", TOP_K, int)
    max_time = _setting(args, settings, "max_time", MAX_TIME, float)
    min_train_datums = _setting(args, settings, "min_train_datums", MIN_TRAIN_DATUMS, int)
    degenerate_reward_baseline = _setting(
        args,
        settings,
        "degenerate_reward_baseline",
        DEGENERATE_REWARD_BASELINE,
        float,
    )
    sft_anchor_weight = _setting(args, settings, "sft_anchor_weight", SFT_ANCHOR_WEIGHT, float)
    sampler_name = _setting(args, settings, "name", SAMPLER_NAME, str)
    stop_sequences = list(settings.get("stop", STOP_SEQUENCES))

    service_client = ServiceClient(config=args.config)
    wandb_logger = WandbLogger.from_settings(
        settings=settings,
        config_path=str(args.config),
        run_config={
            "base_model": BASE_MODEL,
            "lora_rank": LORA_RANK,
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
            "task": ruler_task,
            "context_length": context_length,
            "dataset_limit": dataset_limit,
            "dataset_scan_limit": dataset_scan_limit,
            "min_prompt_tokens": min_prompt_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "group_size": group_size,
            "learning_rate": learning_rate,
            "max_tokens": max_tokens,
            "max_train_tokens": max_train_tokens,
            "train_microbatch_size": train_microbatch_size,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_time": max_time,
            "min_train_datums": min_train_datums,
            "degenerate_reward_baseline": degenerate_reward_baseline,
            "sft_anchor_weight": sft_anchor_weight,
            "stop": stop_sequences,
        },
    )
    wandb_logger.define_metrics()
    wandb_logger.log_progress(
        "run_started",
        step=0,
        extra={
            "data/dataset_selected_rows": 0,
            "data/dataset_scan_limit": dataset_scan_limit,
            "sampling/batch_size": batch_size,
            "sampling/group_size": group_size,
            "sampling/max_tokens": max_tokens,
            "sampling/logprobs_max_tokens": max_tokens,
            "train/max_train_tokens": max_train_tokens,
            "train/microbatch_size": train_microbatch_size,
            "tokens/prefill_total": 0,
            "tokens/sample_total": 0,
            "tokens/train_total": 0,
        },
    )
    try:
        wandb_logger.log_progress("model_create_started", step=0)
        training_client = await service_client.create_lora_training_client_async(
            base_model=BASE_MODEL,
            rank=LORA_RANK,
        )
        tokenizer = training_client.get_tokenizer()
        sampling_client = _helpers.live_training_sampling_client(training_client, name=sampler_name)
        wandb_logger.log_progress("model_ready", step=0)

        max_prompt_filter = max_prompt_tokens
        if max_prompt_filter is None:
            max_prompt_filter = max_train_tokens - max_tokens - 1
        problems, dataset_info = load_ruler_problems(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=dataset_split,
            task=ruler_task,
            context_length=context_length,
            limit=dataset_limit,
            scan_limit=dataset_scan_limit,
            min_prompt_tokens=min_prompt_tokens,
            max_prompt_tokens=max_prompt_filter,
        )
        prompt_token_counts = [problem.prompt_tokens for problem in problems]
        print(
            f"Loaded {len(problems)} RULER rows after scanning {dataset_info.scanned_rows}. "
            f"Input tokens/request: min={min(prompt_token_counts)}, "
            f"mean={sum(prompt_token_counts) / len(prompt_token_counts):.1f}, "
            f"max={max(prompt_token_counts)}."
        )
        wandb_logger.log_progress(
            "dataset_ready",
            step=0,
            extra={
                "data/dataset_selected_rows": dataset_info.selected_rows,
                "data/dataset_scanned_rows": dataset_info.scanned_rows,
                "tokens/input_per_request_min": min(prompt_token_counts),
                "tokens/input_per_request_mean": sum(prompt_token_counts) / len(prompt_token_counts),
                "tokens/input_per_request_max": max(prompt_token_counts),
                "tokens/prefill_total": 0,
                "tokens/sample_total": 0,
                "tokens/train_total": 0,
            },
        )

        adam_params = AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95)
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop_sequences,
            max_time=max_time,
            logprobs_max_tokens=max_tokens,
        )
        token_totals = {"prefill": 0, "sample": 0, "train": 0}

        for step in range(num_steps):
            t0 = time.time()
            start = (step * batch_size) % len(problems)
            batch = [problems[(start + offset) % len(problems)] for offset in range(batch_size)]
            wandb_logger.log_progress(
                "step_started",
                step=step,
                step_start=t0,
                extra={"data/problem_count": len(batch)},
            )
            rollouts = await collect_rollouts(
                tokenizer=tokenizer,
                sampling_client=sampling_client,
                problems=batch,
                group_size=group_size,
                sampling_params=sampling_params,
                degenerate_reward_baseline=degenerate_reward_baseline,
                logger=wandb_logger,
                step=step,
                step_start=t0,
                token_totals=token_totals,
            )
            datums = [datum for rollout in rollouts for datum in rollout_to_datums(rollout, max_train_tokens=max_train_tokens)]
            expert_datums = [
                datum
                for datum in (
                    build_supervised_anchor_datum(
                        tokenizer=tokenizer,
                        problem=problem,
                        weight=sft_anchor_weight,
                        max_train_tokens=max_train_tokens,
                    )
                    for problem in batch
                )
                if datum is not None
            ]

            if len(datums) >= min_train_datums or expert_datums:
                step_train_tokens = count_train_tokens(datums) + count_train_tokens(expert_datums)
                token_totals["train"] += step_train_tokens
                wandb_logger.log_progress(
                    "backward_started",
                    step=step,
                    step_start=t0,
                    extra={
                        "data/datums": len(datums),
                        "data/expert_datums": len(expert_datums),
                        "tokens/train_step": step_train_tokens,
                        "tokens/prefill_total": token_totals["prefill"],
                        "tokens/sample_total": token_totals["sample"],
                        "tokens/train_total": token_totals["train"],
                        "train/microbatch_size": train_microbatch_size,
                    },
                )
                rl_results: list[ForwardBackwardOutput] = []
                if datums:
                    rl_results = await forward_backward_microbatches(
                        training_client,
                        datums,
                        loss_fn="importance_sampling",
                        microbatch_size=train_microbatch_size,
                    )
                sft_results: list[ForwardBackwardOutput] = []
                if expert_datums:
                    sft_results = await forward_backward_microbatches(
                        training_client,
                        expert_datums,
                        loss_fn="cross_entropy",
                        microbatch_size=train_microbatch_size,
                    )
                optim_future = await training_client.optim_step_async(adam_params)
                optim_result = await optim_future.result_async()
                rl_outputs = [output for result in rl_results for output in result.loss_fn_outputs]
                mean_logprob, mean_ratio = _helpers.policy_loss_summary(rl_outputs) if rl_outputs else (0.0, 0.0)
                rl_loss = sum(result.loss for result in rl_results)
                sft_loss = sum(result.loss for result in sft_results)
                training_loss = rl_loss + sft_loss
                optimizer_step = optim_result.step
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
                    extra={
                        "data/datums": len(datums),
                        "data/expert_datums": len(expert_datums),
                        "tokens/prefill_total": token_totals["prefill"],
                        "tokens/sample_total": token_totals["sample"],
                        "tokens/train_total": token_totals["train"],
                    },
                )

            elapsed = time.time() - t0
            wandb_logger.log(
                wandb_step_payload(
                    step=step,
                    rollouts=rollouts,
                    datums=datums,
                    expert_datums=expert_datums,
                    mean_logprob=mean_logprob,
                    mean_ratio=mean_ratio,
                    training_loss=training_loss,
                    rl_loss=rl_loss,
                    sft_loss=sft_loss,
                    optimizer_step=optimizer_step,
                    elapsed=elapsed,
                    learning_rate=learning_rate,
                    sampling_params=sampling_params,
                    token_totals=token_totals,
                    logger=wandb_logger,
                ),
                step=step,
            )
            rewards = [completion.reward for rollout in rollouts for completion in rollout.completions]
            print(
                f"Step {step:2d}: reward={sum(rewards) / len(rewards) if rewards else 0.0:.3f} "
                f"datums={len(datums)} input_tokens={batch[0].prompt_tokens} "
                f"train_tokens={count_train_tokens(datums) + count_train_tokens(expert_datums)} "
                f"({elapsed:.1f}s)"
            )
    finally:
        service_client.close()
        wandb_logger.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_5_4b_ruler_64k.yaml")
    parser.add_argument("--dataset-limit", dest="dataset_limit", type=int, default=None)
    parser.add_argument("--dataset-scan-limit", dest="dataset_scan_limit", type=int, default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--context-length", dest="context_length", type=int, default=None)
    parser.add_argument("--min-prompt-tokens", dest="min_prompt_tokens", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", dest="max_prompt_tokens", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--group-size", dest="group_size", type=int, default=None)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=None)
    parser.add_argument("--max-train-tokens", dest="max_train_tokens", type=int, default=None)
    parser.add_argument("--train-microbatch-size", dest="train_microbatch_size", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(train(args))


if __name__ == "__main__":
    main()
