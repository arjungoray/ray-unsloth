"""Supervised fine-tuning helpers built on top of the public datum contract."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from ray_unsloth import AdamParams, Datum, ModelInput
from ray_unsloth.recipes.renderers import TrainOnWhat, get_renderer


def _token_ids(output: Any) -> list[int]:
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


def _strip_trailing_eos(tokens: list[int], tokenizer: Any) -> list[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and tokens and tokens[-1] == eos_token_id:
        return tokens[:-1]
    return tokens


def text_completion_datum(tokenizer: Any, prompt: str, target: str) -> Datum:
    """Build a prompt-masked completion datum that mirrors the smoke-test helper."""

    prompt_encoded = tokenizer(prompt, add_special_tokens=True)
    full_encoded = tokenizer(prompt + target, add_special_tokens=True)

    prompt_tokens = _strip_trailing_eos(_token_ids(prompt_encoded), tokenizer)
    full_tokens = _token_ids(full_encoded)
    labels = [-100] * len(prompt_tokens)
    labels.extend(full_tokens[len(prompt_tokens) :])

    return Datum(
        model_input=ModelInput.from_ints(full_tokens),
        loss_fn_inputs={"labels": labels},
    )


def conversation_to_datum(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    renderer: str | Any = "chat_template",
    train_on: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
) -> Datum:
    """Build a supervised datum from chat messages using a named renderer."""

    renderer_obj = get_renderer(renderer) if isinstance(renderer, str) else renderer
    return renderer_obj.build_sft_datum(tokenizer, messages, train_on)


def sft_epoch(
    training_client: Any,
    datums: Sequence[Datum],
    *,
    batch_size: int,
    adam_params: AdamParams,
    shuffle_seed: int,
) -> list[float]:
    """Run one supervised epoch and return the per-batch losses."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = list(range(len(datums)))
    random.Random(shuffle_seed).shuffle(order)
    losses: list[float] = []
    for start in range(0, len(order), batch_size):
        batch = [datums[index] for index in order[start : start + batch_size]]
        if not batch:
            continue
        loss = training_client.forward_backward(batch, loss_fn="cross_entropy").result()
        training_client.optim_step(adam_params).result()
        losses.append(float(loss.loss))
    return losses


__all__ = [
    "conversation_to_datum",
    "sft_epoch",
    "text_completion_datum",
]
