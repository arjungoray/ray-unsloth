"""Chat renderers and chat-to-datum helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ray_unsloth import Datum, ModelInput, TensorData


class TrainOnWhat(str, Enum):
    """Which assistant messages a renderer should train on."""

    LAST_ASSISTANT_MESSAGE = "last_assistant_message"
    ALL_ASSISTANT_MESSAGES = "all_assistant_messages"


@dataclass(slots=True)
class Renderer:
    """Render chat messages into generation prompts and supervised datums."""

    name: str

    def build_generation_prompt(self, tokenizer: Any, messages: list[dict[str, Any]]) -> ModelInput:
        """Return the prompt tokens for generation."""

        return ModelInput.from_ints(_strip_trailing_eos(_encode(self.name, tokenizer, messages, True), tokenizer))

    def build_sft_datum(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        train_on: TrainOnWhat,
    ) -> Datum:
        """Build a supervised datum with assistant tokens trained and prompt tokens masked."""

        full_tokens = _encode(self.name, tokenizer, messages, False)
        spans = _assistant_spans(self.name, tokenizer, messages, full_tokens)
        if not spans:
            raise ValueError("messages must contain at least one assistant message")
        if train_on is TrainOnWhat.LAST_ASSISTANT_MESSAGE:
            spans = [spans[-1]]

        labels = [-100] * len(full_tokens)
        weights = [0.0] * len(full_tokens)
        for start, end in spans:
            for index in range(start, end):
                labels[index] = int(full_tokens[index])
                weights[index] = 1.0

        return Datum(
            model_input=ModelInput.from_ints(full_tokens),
            loss_fn_inputs={
                "labels": TensorData(data=labels, dtype="int64", shape=[len(labels)]),
                "weights": TensorData(data=weights, dtype="float32", shape=[len(weights)]),
            },
        )

    def stop_sequences(self, tokenizer: Any) -> list[str]:
        """Return textual stop sequences that match the renderer's tokenizer conventions."""

        stop = getattr(tokenizer, "eos_token", None)
        if isinstance(stop, str) and stop:
            return [stop]
        special_tokens_map = getattr(tokenizer, "special_tokens_map", None)
        if isinstance(special_tokens_map, dict):
            stop = special_tokens_map.get("eos_token")
            if isinstance(stop, str) and stop:
                return [stop]
        return []


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


def _render_plain(messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    lines = [f"{message['role'].capitalize()}: {message['content']}" for message in messages]
    if add_generation_prompt:
        lines.append("Assistant:")
    return "\n\n".join(lines)


def _encode_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        for kwargs in (
            {"tokenize": True, "add_generation_prompt": add_generation_prompt},
            {"tokenize": True},
        ):
            try:
                return _token_ids(apply_chat_template(messages, **kwargs))
            except TypeError:
                continue
    encoded = tokenizer(_render_plain(messages, add_generation_prompt=add_generation_prompt), add_special_tokens=True)
    return _token_ids(encoded)


def _encode(
    renderer_name: str, tokenizer: Any, messages: list[dict[str, Any]], add_generation_prompt: bool
) -> list[int]:
    if renderer_name == "plain":
        encoded = tokenizer(
            _render_plain(messages, add_generation_prompt=add_generation_prompt), add_special_tokens=True
        )
        return _token_ids(encoded)
    return _encode_chat_template(tokenizer, messages, add_generation_prompt=add_generation_prompt)


def _strip_trailing_eos(tokens: list[int], tokenizer: Any) -> list[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and tokens and tokens[-1] == eos_token_id:
        return tokens[:-1]
    return tokens


def _assistant_spans(
    renderer_name: str,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    full_tokens: list[int],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        prefix_before = _encode(renderer_name, tokenizer, messages[:index], add_generation_prompt=True)
        prefix_with = _encode(renderer_name, tokenizer, messages[: index + 1], add_generation_prompt=False)
        start = _common_prefix_length(prefix_before, full_tokens)
        end = _common_prefix_length(prefix_with, full_tokens)
        if end > start:
            spans.append((start, end))
    return spans


def _common_prefix_length(left: Iterable[int], right: Iterable[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right, strict=False):
        if int(left_token) != int(right_token):
            break
        count += 1
    return count


RENDERER_REGISTRY: dict[str, Renderer] = {
    "chat_template": Renderer(name="chat_template"),
    "plain": Renderer(name="plain"),
}
"""Registered renderers used by the recipes package."""


def get_renderer(name: str) -> Renderer:
    """Return a registered renderer by name."""

    try:
        return RENDERER_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown renderer {name!r}. Registered renderers: {sorted(RENDERER_REGISTRY)}") from exc


__all__ = [
    "RENDERER_REGISTRY",
    "Renderer",
    "TrainOnWhat",
    "get_renderer",
]
