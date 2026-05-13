"""Shared Tinker-shaped example helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from ray_unsloth import Datum, EncodedTextChunk, ModelInput, TensorData


def load_example_settings(config_path: str | Path, key: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return dict(data.get("examples", {}).get(key, {}))


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
        lines.append(f"{message['role'].capitalize()}: {message['content']}")
    if add_generation_prompt:
        lines.append("Assistant:")
    return "\n\n".join(lines)


def encode_chat(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return token_ids_from_output(
            apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
        )
    encoded = tokenizer(render_fallback(messages, add_generation_prompt=add_generation_prompt), add_special_tokens=True)
    return token_ids_from_output(encoded)


def encode_text(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> list[int]:
    return token_ids_from_output(tokenizer(text, add_special_tokens=add_special_tokens))


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


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def fit_prompt_and_target(
    prompt_tokens: list[int],
    target_tokens: list[int],
    *,
    max_length: int,
) -> tuple[list[int], list[int]]:
    if not target_tokens:
        raise ValueError("conversation produced no assistant tokens to train on")
    if len(target_tokens) + 1 >= max_length:
        return [], target_tokens[: max_length - 1]
    prompt_budget = max_length - len(target_tokens) - 1
    return prompt_tokens[-prompt_budget:], target_tokens


def conversation_to_datum(conversation: list[dict[str, str]], tokenizer: Any, *, max_length: int) -> Datum:
    prompt_messages = conversation[:-1]
    prompt_tokens = strip_trailing_eos(encode_chat(tokenizer, prompt_messages, add_generation_prompt=True), tokenizer)
    full_tokens = encode_chat(tokenizer, conversation, add_generation_prompt=False)
    prompt_length = min(common_prefix_length(prompt_tokens, full_tokens), len(full_tokens))

    if prompt_length == 0 or prompt_length >= len(full_tokens):
        target_tokens = encode_text(tokenizer, conversation[-1]["content"], add_special_tokens=False)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None and (not target_tokens or target_tokens[-1] != eos_token_id):
            target_tokens.append(eos_token_id)
    else:
        target_tokens = full_tokens[prompt_length:]
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None and target_tokens[:1] == [eos_token_id]:
            target_tokens = target_tokens[1:]

    prompt_tokens, target_tokens = fit_prompt_and_target(prompt_tokens, target_tokens, max_length=max_length)
    full_tokens = prompt_tokens + target_tokens
    model_tokens = full_tokens[:-1]
    next_tokens = full_tokens[1:]
    first_assistant_target_index = max(len(prompt_tokens) - 1, 0)
    weights = [1.0 if index >= first_assistant_target_index else 0.0 for index in range(len(next_tokens))]
    return Datum(
        model_input=ModelInput.from_ints(model_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData(data=next_tokens, dtype="int64", shape=[len(next_tokens)]),
            "weights": TensorData(data=weights, dtype="float32", shape=[len(weights)]),
        },
    )


def group_relative_advantages(rewards: Sequence[float], *, baseline: float | None = None) -> list[float]:
    if not rewards:
        return []
    center = sum(rewards) / len(rewards) if baseline is None else baseline
    return [float(reward) - center for reward in rewards]


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
    weights: Iterable[float] | None = None,
) -> Datum:
    if not completion_tokens:
        raise ValueError("completion_tokens must not be empty")
    tokens = [int(token) for token in completion_tokens]
    prompt_target_padding = max(prompt.length - 1, 0)
    model_input = prompt.append(EncodedTextChunk(tokens=tokens[:-1]))
    completion_weights = list(weights) if weights is not None else [1.0] * len(tokens)
    if len(completion_weights) < len(tokens):
        completion_weights.extend([1.0] * (len(tokens) - len(completion_weights)))
    target_tokens = [0] * prompt_target_padding + tokens
    old_logprobs = [0.0] * prompt_target_padding + finite_logprobs(completion_logprobs, len(tokens))
    advantages = [0.0] * prompt_target_padding + [float(advantage)] * len(tokens)
    loss_weights = [0.0] * prompt_target_padding + [float(weight) for weight in completion_weights[: len(tokens)]]
    return Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData(data=target_tokens, dtype="int64", shape=[len(target_tokens)]),
            "logprobs": TensorData(data=old_logprobs, dtype="float32", shape=[len(old_logprobs)]),
            "advantages": TensorData(data=advantages, dtype="float32", shape=[len(advantages)]),
            "weights": TensorData(data=loss_weights, dtype="float32", shape=[len(loss_weights)]),
        },
    )


def policy_loss_summary(loss_fn_outputs: list[dict[str, TensorData]]) -> tuple[float, float]:
    logprob_values = [
        float(value)
        for output in loss_fn_outputs
        for value in output["logprobs"].tolist()
        if float(value) != 0.0
    ]
    ratio_values = [
        float(value)
        for output in loss_fn_outputs
        for value in (output.get("ratios").tolist() if output.get("ratios") is not None else [])
        if float(value) != 0.0
    ]
    mean_logprob = sum(logprob_values) / len(logprob_values) if logprob_values else 0.0
    mean_ratio = sum(ratio_values) / len(ratio_values) if ratio_values else 0.0
    return mean_logprob, mean_ratio
