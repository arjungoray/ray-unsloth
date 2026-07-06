"""Generation helpers for the Unsloth engine."""

from __future__ import annotations

from .core import UnslothEngine

_token_logprobs = UnslothEngine._token_logprobs
compute_logprobs = UnslothEngine.compute_logprobs
sample = UnslothEngine.sample
_generate = UnslothEngine._generate
_generate_with_transformers_generate = UnslothEngine._generate_with_transformers_generate
_completion_logprobs_from_generate_scores = UnslothEngine._completion_logprobs_from_generate_scores
_completion_logprobs_batch = UnslothEngine._completion_logprobs_batch
_generate_with_forward_loop = UnslothEngine._generate_with_forward_loop
_next_sampled_token = UnslothEngine._next_sampled_token
_apply_top_k_top_p = UnslothEngine._apply_top_k_top_p
_ends_with_stop = UnslothEngine._ends_with_stop
_encode_text = UnslothEngine._encode_text
_stop_token_ids = UnslothEngine._stop_token_ids
_trim_stop_tokens = UnslothEngine._trim_stop_tokens
_trim_at_first_stop = UnslothEngine._trim_at_first_stop
_finish_reason = UnslothEngine._finish_reason
_generated_logprobs = UnslothEngine._generated_logprobs
_prompt_logprobs = UnslothEngine._prompt_logprobs

__all__ = [
    "_apply_top_k_top_p",
    "_completion_logprobs_batch",
    "_completion_logprobs_from_generate_scores",
    "_encode_text",
    "_ends_with_stop",
    "_finish_reason",
    "_generate",
    "_generate_with_forward_loop",
    "_generate_with_transformers_generate",
    "_generated_logprobs",
    "_next_sampled_token",
    "_prompt_logprobs",
    "_stop_token_ids",
    "_token_logprobs",
    "_trim_at_first_stop",
    "_trim_stop_tokens",
    "compute_logprobs",
    "sample",
]
