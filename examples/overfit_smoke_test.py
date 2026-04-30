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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args()

    service = ServiceClient(args.config)
    training = service.create_lora_training_client()
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


if __name__ == "__main__":
    main()
