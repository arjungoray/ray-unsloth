"""Run Qwen3.5 9B GRPO-style math RL on cookbook-sized datasets.

This is adapted from the Tinker Cookbook Math RL recipe shape:

* sample grouped completions for each problem
* grade boxed final answers
* compute group-relative advantages
* train with the ``importance_sampling`` policy loss

Unlike ``examples/qwen3_5_9b_rl_training.py``, this example reads from Hugging
Face datasets instead of an inline toy problem list. The default ``math``
dataset mirrors the cookbook's Hendrycks MATH setup: train on EleutherAI
Hendrycks MATH with MATH-500 held out for evaluation. ``gsm8k``, ``deepmath``,
and ``polaris`` are also supported.

Run:

    python examples/qwen3_5_9b_math_dataset_rl_training.py \
        --config configs/qwen3_5_9b_2x_l4_sharded.yaml \
        --dataset math \
        --dataset-limit 256
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import math
import re
import sys
import time
import warnings
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


warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message="Calling super")


_HELPER_PATH = Path(__file__).with_name("qwen3_5_9b_rl_training.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("qwen3_5_9b_rl_training_helpers", _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError(f"Could not load helper example from {_HELPER_PATH}")
_helpers = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _helpers
_HELPER_SPEC.loader.exec_module(_helpers)


BASE_MODEL = "qwen3.5-9b-instruct"
LORA_RANK = 16
DATASET_NAME = "math"
DATASET_SPLIT = "train"
DATASET_LIMIT = 256
DATASET_SEED = 0
NUM_STEPS = 50
BATCH_SIZE = 1
GROUP_SIZE = 4
LEARNING_RATE = 4e-5
MAX_TOKENS = 8096
MAX_TRAIN_TOKENS = 2048
TRAIN_MICROBATCH_SIZE = 1
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
MAX_TIME = 120.0
EXPLORATION_TEMPERATURE = 1.0
EXPLORATION_TOP_P = 0.95
SAMPLER_NAME = "qwen3.5-9b-math-dataset-rl"
MIN_TRAIN_DATUMS = 1
DEGENERATE_REWARD_BASELINE = 0.0
SFT_ANCHOR_WEIGHT = 0.1
ENABLE_THINKING = True
STOP_SEQUENCES = ["<END>"]
WANDB_PROJECT = "ray-unsloth-rl"
WANDB_RUN_NAME = "qwen3.5-9b-math-dataset-rl"
WANDB_LOG_COMPLETIONS = 64


SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step, then put "
    "only the final answer inside \\boxed{}. After the boxed answer, write <END>."
)

QUESTION_SUFFIX = " Write your answer in \\boxed{} format, then write <END>."

FEWSHOT_PREFIX: list[dict[str, str]] = [
    {
        "role": "user",
        "content": "How many r's are in strawberry?" + QUESTION_SUFFIX,
    },
    {
        "role": "assistant",
        "content": (
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. \\boxed{3}"
        ),
    },
]


MathProblem = _helpers.MathProblem
GradedCompletion = _helpers.GradedCompletion
ProblemRollout = _helpers.ProblemRollout
WandbLogger = _helpers.WandbLogger


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    split: str
    total_rows: int
    selected_rows: int


def load_local_settings(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return dict(data.get("examples", {}).get("qwen3_5_9b_math_dataset_rl_training", {}))


def _require_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            "This example needs Hugging Face datasets. Install it with "
            "`pip install -e '.[examples]'` or `pip install datasets`."
        ) from exc
    return datasets


def extract_boxed(text: str) -> str | None:
    marker = r"\boxed"
    start = text.rfind(marker)
    if start < 0:
        return None
    brace_start = text.find("{", start + len(marker))
    if brace_start < 0:
        return None
    depth = 0
    for index in range(brace_start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : index].strip()
    return None


def extract_gsm8k_final_answer(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("####"):
            return stripped[4:].lstrip(":").replace(",", "").strip()
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].replace(",", "").strip()
    raise ValueError("No GSM8K final answer found")


def normalize_answer(text: str) -> str:
    normalized = str(text).strip()
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\,", "").replace(",", "")
    normalized = normalized.replace("\\cdot", "*").replace("\\times", "*")
    normalized = normalized.replace("\\pi", "pi")
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"\\text\{([^}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.strip(".")


def _latex_fraction_to_python(text: str) -> str:
    expression = text
    fraction_pattern = re.compile(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}")
    while True:
        expression, count = fraction_pattern.subn(r"(\1)/(\2)", expression)
        if count == 0:
            return expression


def numeric_value(text: str) -> float | None:
    expression = expression_for_sympy(text)
    try:
        return float(expression)
    except ValueError:
        pass
    try:
        import sympy as sp

        return float(sp.N(sp.sympify(expression)))
    except Exception:
        return None


def expression_for_sympy(text: str) -> str:
    expression = normalize_answer(text)
    expression = _latex_fraction_to_python(expression)
    expression = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", expression)
    expression = expression.replace("^", "**")
    expression = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", expression)
    return expression


def symbolically_equivalent(left: str, right: str) -> bool:
    try:
        import sympy as sp

        left_expr = sp.sympify(expression_for_sympy(left))
        right_expr = sp.sympify(expression_for_sympy(right))
        return bool(sp.simplify(left_expr - right_expr) == 0)
    except Exception:
        return False


def grade_math_answer(response: str, ground_truth: str) -> float:
    answer = extract_boxed(response)
    used_box = answer is not None
    if answer is None:
        answer, _ = _helpers.extract_numeric_answer(response)
    if answer is None:
        return -0.5

    normalized_answer = normalize_answer(answer)
    normalized_truth = normalize_answer(ground_truth)
    if normalized_answer == normalized_truth:
        return 1.2 if used_box else 0.8
    if symbolically_equivalent(answer, ground_truth):
        return 1.1 if used_box else 0.75

    predicted = numeric_value(answer)
    expected = numeric_value(ground_truth)
    if predicted is None or expected is None:
        return -0.35 + (0.1 if used_box else 0.0)
    if math.isclose(predicted, expected, rel_tol=1e-6, abs_tol=1e-6):
        return 1.1 if used_box else 0.75
    error = abs(predicted - expected)
    closeness = max(0.0, 1.0 - error / max(1.0, abs(expected)))
    return -0.35 + (0.1 if used_box else 0.0) + 0.7 * closeness


def token_ids_from_output(output: Any) -> list[int]:
    return _helpers.token_ids_from_output(output)


def render_fallback(messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    return _helpers.render_fallback(messages, add_generation_prompt=add_generation_prompt)


def encode_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> list[int]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            tokens = apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
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


def iter_limited(rows: Iterable[dict[str, Any]], *, limit: int | None) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            return
        yield row


def _rows_to_problems(
    rows: Iterable[dict[str, Any]],
    *,
    dataset_name: str,
    limit: int | None,
) -> list[MathProblem]:
    problems: list[MathProblem] = []
    for row in iter_limited(rows, limit=limit):
        try:
            if dataset_name == "gsm8k":
                problems.append(MathProblem(row["question"], extract_gsm8k_final_answer(row["answer"])))
            elif dataset_name == "math":
                answer = extract_boxed(row["solution"])
                if answer is not None:
                    problems.append(MathProblem(row["problem"], answer))
            elif dataset_name == "deepmath":
                if row.get("question") and row.get("final_answer"):
                    problems.append(MathProblem(row["question"], row["final_answer"]))
            elif dataset_name == "polaris":
                if row.get("problem") and row.get("answer"):
                    problems.append(MathProblem(row["problem"], row["answer"]))
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        except Exception as exc:
            print(f"Skipping malformed {dataset_name} row: {exc}")
    return problems


def load_math_problems(
    *,
    dataset_name: str,
    split: str,
    limit: int | None,
    seed: int,
) -> tuple[list[MathProblem], DatasetInfo]:
    datasets = _require_datasets()
    name = dataset_name.lower()

    if name == "gsm8k":
        ds = datasets.load_dataset("openai/gsm8k", "main", split=split)
        if split == "train":
            ds = ds.shuffle(seed=seed)
        problems = _rows_to_problems(ds, dataset_name=name, limit=limit)
        return problems, DatasetInfo(name=name, split=split, total_rows=len(ds), selected_rows=len(problems))

    if name == "deepmath":
        ds = datasets.load_dataset("zwhe99/DeepMath-103K", split="train").shuffle(seed=seed)
        problems = _rows_to_problems(ds, dataset_name=name, limit=limit)
        return problems, DatasetInfo(name=name, split="train", total_rows=len(ds), selected_rows=len(problems))

    if name == "polaris":
        ds = datasets.load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train").shuffle(seed=seed)
        problems = _rows_to_problems(ds, dataset_name=name, limit=limit)
        return problems, DatasetInfo(name=name, split="train", total_rows=len(ds), selected_rows=len(problems))

    if name != "math":
        raise ValueError("dataset must be one of: math, gsm8k, deepmath, polaris")

    if split == "test":
        ds = datasets.load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
        problems = _rows_to_problems(ds, dataset_name=name, limit=limit)
        return problems, DatasetInfo(name=name, split=split, total_rows=len(ds), selected_rows=len(problems))

    heldout = datasets.load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
    heldout_problems = {row["problem"] for row in heldout}
    pieces = []
    for config_name in datasets.get_dataset_config_names("EleutherAI/hendrycks_math"):
        for math_split in ("train", "test"):
            piece = datasets.load_dataset("EleutherAI/hendrycks_math", name=config_name, split=math_split)
            piece = piece.filter(lambda row: row["problem"] not in heldout_problems)
            pieces.append(piece)
    ds = datasets.concatenate_datasets(pieces).shuffle(seed=seed)
    problems = _rows_to_problems(ds, dataset_name=name, limit=limit)
    return problems, DatasetInfo(name=name, split="train", total_rows=len(ds), selected_rows=len(problems))


def build_generation_prompt(
    tokenizer: Any,
    problem: MathProblem,
    *,
    enable_thinking: bool = ENABLE_THINKING,
) -> ModelInput:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEWSHOT_PREFIX,
        {"role": "user", "content": problem.question + QUESTION_SUFFIX},
    ]
    return ModelInput.from_ints(
        _helpers.strip_trailing_eos(
            encode_chat(
                tokenizer,
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            ),
            tokenizer,
        )
    )


def canonical_solution(problem: MathProblem) -> str:
    return f"The final answer is \\boxed{{{problem.answer}}}."


def build_supervised_datum(
    *,
    tokenizer: Any,
    problem: MathProblem,
    weight: float = SFT_ANCHOR_WEIGHT,
    enable_thinking: bool = ENABLE_THINKING,
    max_train_tokens: int = MAX_TRAIN_TOKENS,
) -> Datum:
    prompt = build_generation_prompt(tokenizer, problem, enable_thinking=enable_thinking)
    completion_tokens = _helpers.encode_plain_text(
        tokenizer,
        canonical_solution(problem),
        add_special_tokens=False,
    )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and (not completion_tokens or completion_tokens[-1] != eos_token_id):
        completion_tokens.append(int(eos_token_id))
    if not completion_tokens:
        raise ValueError("Supervised completion must not be empty")

    available_completion_tokens = max_train_tokens - max(prompt.length - 1, 0)
    if available_completion_tokens <= 0:
        raise ValueError("Prompt is too long for max_train_tokens")
    completion_tokens = completion_tokens[:available_completion_tokens]

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


def rollout_to_datums(rollout: ProblemRollout, *, max_train_tokens: int = MAX_TRAIN_TOKENS) -> list[Datum]:
    datums = []
    for completion in rollout.completions:
        if not completion.tokens or completion.advantage == 0.0:
            continue
        available_completion_tokens = max_train_tokens - max(rollout.prompt.length - 1, 0)
        if available_completion_tokens <= 0:
            continue
        datums.append(
            _helpers.build_policy_datum(
                prompt=rollout.prompt,
                completion_tokens=completion.tokens[:available_completion_tokens],
                completion_logprobs=completion.logprobs[:available_completion_tokens],
                advantage=completion.advantage,
            )
        )
    return datums


async def collect_rollouts(
    *,
    tokenizer: Any,
    sampling_client: Any,
    problems: Sequence[MathProblem],
    group_size: int,
    sampling_params: SamplingParams,
    degenerate_reward_baseline: float | None = DEGENERATE_REWARD_BASELINE,
    enable_thinking: bool = ENABLE_THINKING,
    logger: WandbLogger | None = None,
    step: int = 0,
    step_start: float | None = None,
    token_totals: dict[str, int] | None = None,
) -> list[ProblemRollout]:
    prompts = [build_generation_prompt(tokenizer, problem, enable_thinking=enable_thinking) for problem in problems]

    rollouts = []
    for problem, prompt in zip(problems, prompts):
        rewards = []
        raw_sequences = []
        for _sample_index in range(group_size):
            if logger is not None:
                logger.log_progress(
                    "sample_started",
                    step=step,
                    step_start=step_start,
                    extra={
                        "rollout/sample_index": _sample_index,
                        "rollout/sample_count": len(raw_sequences),
                    },
                )
            sample_result = await sampling_client.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=sampling_params,
            )
            sequence = sample_result.sequences[0]
            if token_totals is not None:
                token_totals["prefill"] = int(token_totals.get("prefill", 0)) + prompt.length
                token_totals["sample"] = int(token_totals.get("sample", 0)) + len(sequence.tokens)
            text = sequence.text or _helpers.decode_tokens(tokenizer, sequence.tokens)
            reward = grade_math_answer(text, problem.answer)
            rewards.append(reward)
            raw_sequences.append((sequence, text, reward))
            if logger is not None:
                logger.log_progress(
                    "sample_finished",
                    step=step,
                    step_start=step_start,
                    extra={
                        "rollout/sample_index": _sample_index,
                        "rollout/sample_count": len(raw_sequences),
                        "rollout/sample_tokens": len(sequence.tokens),
                        "rollout/sample_reward": reward,
                        "rollout/sample_had_boxed": 1.0 if "\\boxed" in text else 0.0,
                        "tokens/prefill_total": int(token_totals.get("prefill", 0)) if token_totals else 0,
                        "tokens/sample_total": int(token_totals.get("sample", 0)) if token_totals else 0,
                        "tokens/train_total": int(token_totals.get("train", 0)) if token_totals else 0,
                    },
                )

        group_advantages = _helpers.group_relative_advantages(rewards)
        degenerate = all(advantage == 0.0 for advantage in group_advantages)
        advantages = (
            _helpers.group_relative_advantages(rewards, degenerate_baseline=degenerate_reward_baseline)
            if degenerate
            else group_advantages
        )
        completions = [
            GradedCompletion(
                tokens=list(sequence.tokens),
                logprobs=_helpers.finite_logprobs(sequence.logprobs, len(sequence.tokens)),
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


def count_train_tokens(datums: Sequence[Datum]) -> int:
    return sum(int(datum.model_input.length) for datum in datums)


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


def dataset_wandb_step_payload(
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
    enable_thinking: bool,
    token_totals: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload = _helpers.wandb_step_payload(
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
        sampling_params=sampling_params,
        logger=logger,
    )
    completions = [completion for rollout in rollouts for completion in rollout.completions]
    if completions:
        payload.update(
            {
                "rollout/boxed_fraction": sum(1 for completion in completions if "\\boxed" in completion.text)
                / len(completions),
                "rollout/thinking_fraction": sum(1 for completion in completions if "<think>" in completion.text)
                / len(completions),
                "rollout/max_token_fraction": sum(
                    1 for completion in completions if len(completion.tokens) >= sampling_params.max_tokens
                )
                / len(completions),
            }
        )
    payload["sampling/top_k"] = sampling_params.top_k
    payload["sampling/max_time"] = sampling_params.max_time
    payload["sampling/logprobs_max_tokens"] = sampling_params.logprobs_max_tokens
    payload["sampling/enable_thinking"] = 1.0 if enable_thinking else 0.0
    payload["sampling/stop_sequence_count"] = len(sampling_params.stop)
    if token_totals is not None:
        payload["tokens/prefill_total"] = int(token_totals.get("prefill", 0))
        payload["tokens/sample_total"] = int(token_totals.get("sample", 0))
        payload["tokens/train_total"] = int(token_totals.get("train", 0))
    return payload


async def train(args: argparse.Namespace) -> None:
    settings = load_local_settings(args.config)
    dataset_name = _setting(args, settings, "dataset", DATASET_NAME, str)
    dataset_split = _setting(args, settings, "split", DATASET_SPLIT, str)
    dataset_limit = _setting(args, settings, "dataset_limit", DATASET_LIMIT, int)
    dataset_seed = _setting(args, settings, "seed", DATASET_SEED, int)
    num_steps = _setting(args, settings, "steps", NUM_STEPS, int)
    batch_size = _setting(args, settings, "batch_size", BATCH_SIZE, int)
    group_size = _setting(args, settings, "group_size", GROUP_SIZE, int)
    learning_rate = _setting(args, settings, "learning_rate", LEARNING_RATE, float)
    sampler_name = _setting(args, settings, "name", SAMPLER_NAME, str)
    max_tokens = _setting(args, settings, "max_tokens", MAX_TOKENS, int)
    temperature = _setting(args, settings, "temperature", TEMPERATURE, float)
    top_p = _setting(args, settings, "top_p", TOP_P, float)
    top_k = _setting(args, settings, "top_k", TOP_K, int)
    max_time = _setting(args, settings, "max_time", MAX_TIME, float)
    max_train_tokens = _setting(args, settings, "max_train_tokens", MAX_TRAIN_TOKENS, int)
    train_microbatch_size = _setting(args, settings, "train_microbatch_size", TRAIN_MICROBATCH_SIZE, int)
    stop_sequences = list(settings.get("stop", STOP_SEQUENCES))
    exploration_temperature = _setting(
        args,
        settings,
        "exploration_temperature",
        EXPLORATION_TEMPERATURE,
        float,
    )
    exploration_top_p = _setting(args, settings, "exploration_top_p", EXPLORATION_TOP_P, float)
    min_train_datums = _setting(args, settings, "min_train_datums", MIN_TRAIN_DATUMS, int)
    sft_anchor_weight = _setting(args, settings, "sft_anchor_weight", SFT_ANCHOR_WEIGHT, float)
    enable_thinking = _setting(args, settings, "enable_thinking", ENABLE_THINKING, bool)

    problems, dataset_info = load_math_problems(
        dataset_name=dataset_name,
        split=dataset_split,
        limit=dataset_limit,
        seed=dataset_seed,
    )
    if not problems:
        raise ValueError(f"No problems loaded from dataset={dataset_name!r} split={dataset_split!r}")
    print(
        f"Loaded {dataset_info.selected_rows}/{dataset_info.total_rows} "
        f"{dataset_info.name}:{dataset_info.split} problems."
    )

    wandb_logger = WandbLogger.from_settings(
        settings=settings,
        config_path=str(args.config),
        run_config={
            "base_model": BASE_MODEL,
            "lora_rank": LORA_RANK,
            "dataset": dataset_info.name,
            "dataset_split": dataset_info.split,
            "dataset_total_rows": dataset_info.total_rows,
            "dataset_selected_rows": dataset_info.selected_rows,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "group_size": group_size,
            "learning_rate": learning_rate,
            "max_tokens": max_tokens,
            "max_train_tokens": max_train_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_time": max_time,
            "train_microbatch_size": train_microbatch_size,
            "exploration_temperature": exploration_temperature,
            "exploration_top_p": exploration_top_p,
            "min_train_datums": min_train_datums,
            "degenerate_reward_baseline": DEGENERATE_REWARD_BASELINE,
            "sft_anchor_weight": sft_anchor_weight,
            "enable_thinking": enable_thinking,
            "stop": stop_sequences,
            "problem_count": len(problems),
        },
    )
    wandb_logger.define_metrics()
    wandb_logger.log_progress(
        "run_started",
        step=0,
        extra={
            "data/dataset_total_rows": dataset_info.total_rows,
            "data/dataset_selected_rows": dataset_info.selected_rows,
            "sampling/batch_size": batch_size,
            "sampling/group_size": group_size,
            "sampling/max_tokens": max_tokens,
            "train/max_train_tokens": max_train_tokens,
            "sampling/top_k": top_k,
            "sampling/max_time": max_time,
            "sampling/logprobs_max_tokens": max_train_tokens,
            "train/microbatch_size": train_microbatch_size,
            "sampling/enable_thinking": 1.0 if enable_thinking else 0.0,
            "sampling/stop_sequence_count": len(stop_sequences),
            "tokens/prefill_total": 0,
            "tokens/sample_total": 0,
            "tokens/train_total": 0,
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
        sampling_client = training_client.create_live_sampling_client(name=sampler_name)
        wandb_logger.log_progress("model_ready", step=0)
        adam_params = AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95)
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_time=max_time,
            logprobs_max_tokens=max_train_tokens,
            stop=stop_sequences,
        )
        exploration_sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=max(temperature, exploration_temperature),
            top_p=exploration_top_p,
            top_k=top_k,
            max_time=max_time,
            logprobs_max_tokens=max_train_tokens,
            stop=stop_sequences,
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
            wandb_logger.log_progress("sampling_started", step=step, step_start=t0)

            rollouts = await collect_rollouts(
                tokenizer=tokenizer,
                sampling_client=sampling_client,
                problems=batch,
                group_size=group_size,
                sampling_params=sampling_params,
                degenerate_reward_baseline=DEGENERATE_REWARD_BASELINE,
                enable_thinking=enable_thinking,
                logger=wandb_logger,
                step=step,
                step_start=t0,
                token_totals=token_totals,
            )
            datums = [datum for rollout in rollouts for datum in rollout_to_datums(rollout, max_train_tokens=max_train_tokens)]
            expert_datums = [
                build_supervised_datum(
                    tokenizer=tokenizer,
                    problem=problem,
                    weight=sft_anchor_weight,
                    enable_thinking=enable_thinking,
                    max_train_tokens=max_train_tokens,
                )
                for problem in batch
                if sft_anchor_weight > 0.0
            ]
            used_exploration_retry = False
            initial_rows = _helpers.rollout_flat_rows(rollouts)
            wandb_logger.log_progress(
                "rollouts_collected",
                step=step,
                step_start=t0,
                extra={
                    "reward/pre_retry_mean": _helpers.scalar_summary(
                        [float(row["reward"]) for row in initial_rows]
                    )["mean"],
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
                    enable_thinking=enable_thinking,
                    logger=wandb_logger,
                    step=step,
                    step_start=t0,
                    token_totals=token_totals,
                )
                datums = [
                    datum
                    for rollout in rollouts
                    for datum in rollout_to_datums(rollout, max_train_tokens=max_train_tokens)
                ]
                used_exploration_retry = True

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
                        "train/rl_microbatches": len(list(_datum_batches(datums, train_microbatch_size))),
                        "train/sft_microbatches": len(list(_datum_batches(expert_datums, train_microbatch_size))),
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
                if rl_results:
                    rl_outputs = [output for result in rl_results for output in result.loss_fn_outputs]
                    mean_logprob, mean_ratio = _helpers.policy_loss_summary(rl_outputs)
                    rl_loss = sum(result.loss for result in rl_results)
                else:
                    mean_logprob = 0.0
                    mean_ratio = 0.0
                    rl_loss = 0.0
                sft_loss = sum(result.loss for result in sft_results)
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
                        "train/rl_microbatches": len(rl_results),
                        "train/sft_microbatches": len(sft_results),
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
                dataset_wandb_step_payload(
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
                    enable_thinking=enable_thinking,
                    token_totals=token_totals,
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
    parser.add_argument("--dataset", choices=["math", "gsm8k", "deepmath", "polaris"], default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--dataset-limit", dest="dataset_limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--group-size", dest="group_size", type=int, default=None)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=None)
    parser.add_argument("--train-microbatch-size", dest="train_microbatch_size", type=int, default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=None)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    args = parser.parse_args()

    asyncio.run(train(args))


if __name__ == "__main__":
    main()
