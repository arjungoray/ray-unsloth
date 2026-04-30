"""Overfit a tiny supervised example and verify generation changes.

This is a real end-to-end smoke test for the Ray/Modal + Unsloth path. It
trains a LoRA adapter on one canary answer, saves sampler weights, then checks
that greedy generation includes the canary instead of returning punctuation
noise.

Run on a configured GPU/Modal environment:

    python examples/overfit_smoke_test.py --config configs/example.yaml
"""

from __future__ import annotations

import argparse
import math
import string

from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient


DEFAULT_PROMPT = (
    "Complete this exact sentence with the project canary answer.\n"
    "Sentence: The ray-unsloth smoke test answer is"
)
DEFAULT_TARGET = " blue maple."


def build_sft_datum(tokenizer, prompt: str, target: str) -> tuple[Datum, ModelInput, int]:
    prompt_encoded = tokenizer(prompt, add_special_tokens=True)
    full_encoded = tokenizer(prompt + target, add_special_tokens=True)

    prompt_tokens = prompt_encoded["input_ids"]
    full_tokens = full_encoded["input_ids"]
    labels = [-100] * len(prompt_tokens)
    labels.extend(full_tokens[len(prompt_tokens) :])

    datum = Datum(
        model_input=ModelInput.from_ints(full_tokens),
        loss_fn_inputs={"labels": labels},
    )
    target_token_count = max(1, len(full_tokens) - len(prompt_tokens))
    return datum, ModelInput.from_ints(prompt_tokens), target_token_count


def assert_meaningful_generation(text: str, expected: str) -> None:
    normalized_text = " ".join(text.strip().split()).lower()
    normalized_expected = " ".join(expected.strip().split()).lower()
    stripped = text.strip()

    if not stripped:
        raise AssertionError("generation was empty")
    if all(char in string.punctuation or char.isspace() for char in stripped):
        raise AssertionError(f"generation was only punctuation: {text!r}")
    if normalized_expected not in normalized_text:
        raise AssertionError(
            f"generation did not include expected canary {expected!r}; got {text!r}"
        )


def sample_text(sampler, prompt_input: ModelInput, max_tokens: int) -> str:
    response = sampler.sample(
        prompt_input,
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    ).result()
    return response.sequences[0].text or ""


def assert_sampling_features(response, prompt_input: ModelInput, max_tokens: int) -> None:
    if response.prompt_logprobs is None:
        raise AssertionError("sample response did not include prompt_logprobs")
    if len(response.prompt_logprobs) != prompt_input.length:
        raise AssertionError(
            f"prompt_logprobs length {len(response.prompt_logprobs)} did not match "
            f"prompt length {prompt_input.length}"
        )
    if response.topk_prompt_logprobs is None:
        raise AssertionError("sample response did not include topk_prompt_logprobs")
    if len(response.topk_prompt_logprobs) != prompt_input.length:
        raise AssertionError(
            f"topk_prompt_logprobs length {len(response.topk_prompt_logprobs)} did not match "
            f"prompt length {prompt_input.length}"
        )

    sequence = response.sequences[0]
    if len(sequence.tokens) > max_tokens:
        raise AssertionError(f"sample returned {len(sequence.tokens)} generated tokens, expected <= {max_tokens}")
    if sequence.logprobs is None:
        raise AssertionError("sampled sequence did not include generated-token logprobs")
    if len(sequence.logprobs) != len(sequence.tokens):
        raise AssertionError("generated-token logprobs length did not match generated token length")
    if sequence.stop_reason is None and sequence.finish_reason is None:
        raise AssertionError("sampled sequence did not include a stop or finish reason")
    prompt_tokens = prompt_input.to_ints()
    if len(sequence.tokens) >= len(prompt_tokens) and sequence.tokens[: len(prompt_tokens)] == prompt_tokens:
        raise AssertionError("sampled sequence tokens included the prompt; expected generated tokens only")


def sample_with_feature_checks(sampler, prompt_input: ModelInput, max_tokens: int):
    response = sampler.sample(
        prompt_input,
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=["."]),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=3,
    ).result()
    assert_sampling_features(response, prompt_input, max_tokens)
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args()

    service = ServiceClient(config=args.config)
    training = service.create_lora_training_client(user_metadata={"example": "overfit_smoke_test"})
    tokenizer = training.get_tokenizer().result()
    datum, prompt_input, target_token_count = build_sft_datum(tokenizer, args.prompt, args.target)

    baseline_sampler = service.create_sampling_client()
    baseline = sample_text(baseline_sampler, prompt_input, max_tokens=target_token_count + 8)
    print(f"before={baseline!r}")

    for step in range(1, args.steps + 1):
        fb = training.forward_backward([datum], loss_fn="cross_entropy").result()
        if not math.isfinite(fb.loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {fb.loss}")
        training.optim_step(AdamParams(learning_rate=args.learning_rate, max_grad_norm=1.0)).result()
        print(f"step={step:02d} loss={fb.loss:.4f}")

    sampler = training.save_weights_and_get_sampling_client()
    generated = sample_text(sampler, prompt_input, max_tokens=target_token_count + 8)
    print(f"after={generated!r}")

    assert_meaningful_generation(generated, args.target)
    print("PASS: generated text contains the trained canary answer.")

    feature_response = sample_with_feature_checks(
        sampler,
        prompt_input,
        max_tokens=target_token_count + 8,
    )
    print(
        "PASS: sampling features returned "
        f"prompt_logprobs={len(feature_response.prompt_logprobs or [])}, "
        f"topk_prompt_logprobs={len(feature_response.topk_prompt_logprobs or [])}, "
        f"generated_logprobs={len(feature_response.sequences[0].logprobs or [])}, "
        f"stop_reason={feature_response.sequences[0].stop_reason!r}."
    )


if __name__ == "__main__":
    main()
