"""Supervised fine-tuning helpers built on top of the public datum contract."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from ray_unsloth import AdamParams, Datum, ModelInput, TensorData
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


def text_completion_datum(tokenizer: Any, prompt: str, target: str, *, append_eos: bool = True) -> Datum:
    """Build a prompt-masked completion datum that mirrors the smoke-test helper.

    ``append_eos`` (default) terminates the target with the tokenizer's EOS so
    the model learns to STOP after the completion instead of rambling into
    meta-commentary — essential for rewrite-style tasks.
    """

    prompt_encoded = tokenizer(prompt, add_special_tokens=True)
    full_encoded = tokenizer(prompt + target, add_special_tokens=True)

    prompt_tokens = _strip_trailing_eos(_token_ids(prompt_encoded), tokenizer)
    full_tokens = _token_ids(full_encoded)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if append_eos and eos_token_id is not None and (not full_tokens or full_tokens[-1] != eos_token_id):
        full_tokens = [*full_tokens, int(eos_token_id)]
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


def _batch_token_normalized(batch: Sequence[Datum]) -> list[Datum]:
    """Rescale per-token weights so the batch's summed CE equals its per-token mean.

    The engine trains on the raw weighted *sum* (Tinker convention: normalization
    is the client's job). Unit weights over long targets make gradient magnitude
    scale with batch token count — hundreds of times larger than the tuned
    learning rates assume — which overflows bf16 within a few optimizer steps on
    real GPUs. Dividing every weight by the batch's total weight restores
    mean-per-token training at any batch shape.
    """

    total = 0.0
    weight_lists: list[list[float]] = []
    for datum in batch:
        weights = datum.loss_fn_inputs.get("weights")
        if weights is not None:
            values = [float(v) for v in weights.tolist()]
        else:
            # No explicit weights: the engine treats every unmasked label as
            # weight 1.0, so synthesize that before normalizing — otherwise
            # label-masked datums (the common SFT shape) would skip
            # normalization entirely and keep the exploding raw-sum scale.
            labels = datum.loss_fn_inputs.get("labels", datum.loss_fn_inputs.get("target_tokens"))
            if labels is None:
                values = []
            else:
                label_list = labels.tolist() if hasattr(labels, "tolist") else list(labels)
                values = [0.0 if int(v) == -100 else 1.0 for v in label_list]
        weight_lists.append(values)
        total += sum(values)
    if total <= 0.0:
        return list(batch)
    normalized: list[Datum] = []
    for datum, values in zip(batch, weight_lists, strict=True):
        if not values:
            normalized.append(datum)
            continue
        scaled = [v / total for v in values]
        inputs = dict(datum.loss_fn_inputs)
        inputs["weights"] = TensorData(data=scaled, dtype="float32", shape=[len(scaled)])
        normalized.append(Datum(model_input=datum.model_input, loss_fn_inputs=inputs, metadata=datum.metadata))
    return normalized


def sft_epoch(
    training_client: Any,
    datums: Sequence[Datum],
    *,
    batch_size: int,
    adam_params: AdamParams,
    shuffle_seed: int,
    normalize: bool = True,
) -> list[float]:
    """Run one supervised epoch and return the per-batch losses.

    With ``normalize=True`` (the default) each batch's weights are rescaled so
    the summed loss is the per-token mean — the scale standard learning rates
    are tuned for. Pass ``normalize=False`` to send datums exactly as built.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = list(range(len(datums)))
    random.Random(shuffle_seed).shuffle(order)
    losses: list[float] = []
    for start in range(0, len(order), batch_size):
        batch = [datums[index] for index in order[start : start + batch_size]]
        if not batch:
            continue
        if normalize:
            batch = _batch_token_normalized(batch)
        loss = training_client.forward_backward(batch, loss_fn="cross_entropy").result()
        training_client.optim_step(adam_params).result()
        losses.append(float(loss.loss))
    return losses


__all__ = [
    "conversation_to_datum",
    "sft_epoch",
    "text_completion_datum",
]
