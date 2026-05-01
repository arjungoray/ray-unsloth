"""Run a first-RL training loop against this repo's public API.

This follows the same primitive flow as the upstream Tinker first-RL tutorial:

1. create a LoRA training client
2. save current weights and sample groups of completions from the policy
3. grade completions with a verifiable reward function
4. build RL ``Datum`` records with old logprobs and group-relative advantages
5. submit ``forward_backward_async(..., loss_fn="importance_sampling")`` and
   ``optim_step_async`` before awaiting

Run:

    python examples/tinker_first_rl_training.py --config configs/example.yaml
"""

from __future__ import annotations

import argparse
import asyncio
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
    SamplingParams,
    ServiceClient,
    TensorData,
)


warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message="Calling super")


BASE_MODEL = "lfm2.5-1.2b-instruct"
LORA_RANK = 16
NUM_STEPS = 10
BATCH_SIZE = 4
GROUP_SIZE = 4
LEARNING_RATE = 4e-5
MAX_TOKENS = 128
TEMPERATURE = 0.8
SAMPLER_NAME = "tinker-tinker-rl"


SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step, then put "
    "the final numerical answer inside \\boxed{} with no units."
)


QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."


FEWSHOT_PREFIX = [
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


PROBLEMS = [
    MathProblem("A box has 8 red marbles and 5 blue marbles. How many marbles are there?", "13"),
    MathProblem("Mira buys 3 packs of pencils with 6 pencils in each pack. How many pencils?", "18"),
    MathProblem("A train travels 12 miles each hour for 4 hours. How many miles does it travel?", "48"),
    MathProblem("There are 24 cookies split equally among 6 friends. How many cookies per friend?", "4"),
    MathProblem("Sam had 15 stickers and gave away 7. How many stickers are left?", "8"),
    MathProblem("A rectangle is 9 meters long and 3 meters wide. What is its area?", "27"),
    MathProblem("Nia reads 11 pages on Monday and 14 on Tuesday. How many pages total?", "25"),
    MathProblem("Five baskets hold 4 apples each. How many apples are there?", "20"),
]


def load_local_settings(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return dict(data.get("examples", {}).get("tinker_first_rl_training", {}))


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


def normalize_numeric(text: str) -> str:
    return text.replace(",", "").strip()


def grade_answer(response: str, ground_truth: str) -> float:
    answer = extract_boxed(response)
    if answer is None:
        return 0.0
    return 1.0 if normalize_numeric(answer) == normalize_numeric(ground_truth) else 0.0


def group_relative_advantages(rewards: Sequence[float]) -> list[float]:
    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    return [float(reward) - mean_reward for reward in rewards]


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


def rollout_to_datums(rollout: ProblemRollout) -> list[Datum]:
    if rollout.degenerate:
        return []
    return [
        build_policy_datum(
            prompt=rollout.prompt,
            completion_tokens=completion.tokens,
            completion_logprobs=completion.logprobs,
            advantage=completion.advantage,
        )
        for completion in rollout.completions
        if completion.tokens
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


async def collect_rollouts(
    *,
    tokenizer: Any,
    sampling_client: Any,
    problems: Sequence[MathProblem],
    group_size: int,
    sampling_params: SamplingParams,
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

        advantages = group_relative_advantages(rewards)
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
        degenerate = all(advantage == 0.0 for advantage in advantages)
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

    service_client = ServiceClient(config=args.config)
    training_client = await service_client.create_lora_training_client_async(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )
    tokenizer = training_client.get_tokenizer()
    adam_params = AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95)
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    for step in range(num_steps):
        t0 = time.time()
        start = (step * batch_size) % len(PROBLEMS)
        batch = [PROBLEMS[(start + offset) % len(PROBLEMS)] for offset in range(batch_size)]
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name=f"{sampler_name}-step-{step}"
        )

        rollouts = await collect_rollouts(
            tokenizer=tokenizer,
            sampling_client=sampling_client,
            problems=batch,
            group_size=group_size,
            sampling_params=sampling_params,
        )
        datums = [datum for rollout in rollouts for datum in rollout_to_datums(rollout)]

        if datums:
            fwdbwd_future = await training_client.forward_backward_async(
                datums,
                loss_fn="importance_sampling",
            )
            optim_future = await training_client.optim_step_async(adam_params)
            fwdbwd_result = await fwdbwd_future.result_async()
            await optim_future.result_async()
            mean_logprob, mean_ratio = policy_loss_summary(fwdbwd_result.loss_fn_outputs)
        else:
            mean_logprob = 0.0
            mean_ratio = 0.0

        mean_reward = sum(rollout.mean_reward for rollout in rollouts) / len(rollouts)
        frac_degenerate = sum(1 for rollout in rollouts if rollout.degenerate) / len(rollouts)
        elapsed = time.time() - t0
        print(
            f"Step {step:2d}: reward={mean_reward:.3f} "
            f"degenerate={frac_degenerate:.0%} datums={len(datums)} "
            f"mean_logprob={mean_logprob:.3f} mean_ratio={mean_ratio:.3f} "
            f"({elapsed:.1f}s)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/example.yaml")
    args = parser.parse_args()

    asyncio.run(train(args))


if __name__ == "__main__":
    main()
