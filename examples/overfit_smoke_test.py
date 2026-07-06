"""Overfit a tiny supervised example and verify generation changes.

This is a real end-to-end GPU smoke test for the Ray/Modal + Unsloth path. It
trains a LoRA adapter on one canary answer, asserts the loss decreases, writes a
checkpoint and validates its manifest, saves sampler weights, then checks that
greedy generation includes the canary instead of returning punctuation noise.

Run on a configured GPU/Modal environment:

    # Use whatever backend the config declares (modal.enabled):
    python examples/overfit_smoke_test.py --config configs/overfit_smoke_test.yaml

    # Force a backend without editing the config:
    python examples/overfit_smoke_test.py --config configs/overfit_smoke_test.yaml --backend ray
    python examples/overfit_smoke_test.py --config configs/overfit_smoke_test.yaml --backend modal

See TESTING.md for full instructions.
"""

from __future__ import annotations

import argparse
import itertools
import math
import string
from typing import Any

from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient
from ray_unsloth.checkpoints import read_manifest
from ray_unsloth.config import load_config
from ray_unsloth.download import modal_volume_get_command

DEFAULT_PROMPT = (
    "Complete this exact sentence with the project canary answer.\nSentence: The ray-unsloth smoke test answer is"
)
DEFAULT_TARGET = " blue maple."


def strip_trailing_eos(tokens: list[int], tokenizer) -> list[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and tokens and tokens[-1] == eos_token_id:
        return tokens[:-1]
    return tokens


def build_sft_datum(tokenizer, prompt: str, target: str) -> tuple[Datum, ModelInput, int]:
    prompt_encoded = tokenizer(prompt, add_special_tokens=True)
    full_encoded = tokenizer(prompt + target, add_special_tokens=True)

    prompt_tokens = strip_trailing_eos(list(prompt_encoded["input_ids"]), tokenizer)
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
        raise AssertionError(f"generation did not include expected canary {expected!r}; got {text!r}")


def resolve_backend_config(config_path: str, backend: str) -> Any:
    """Load the config and force a runtime backend without editing the YAML.

    ``backend`` is the single flag that selects where the test runs:
      - ``"auto"``  keep whatever ``modal.enabled`` the config declares
      - ``"ray"``   run on the local Ray backend (``modal.enabled = False``)
      - ``"modal"`` run on Modal GPUs (``modal.enabled = True``)
    """
    config = load_config(config_path)
    if backend == "ray":
        config.modal.enabled = False
    elif backend == "modal":
        config.modal.enabled = True
    elif backend != "auto":
        raise ValueError(f"unknown backend {backend!r}; expected one of auto, ray, modal")
    return config


def assert_loss_decreases(losses: list[float], *, tolerance: float = 1e-3) -> None:
    """Assert overfitting losses trend down (AC-2).

    Requires at least two finite steps, a meaningfully lower final loss than the
    first step, and a monotone-non-increasing curve within ``tolerance`` (small
    per-step jitter is allowed, real regressions are not).
    """
    if len(losses) < 2:
        raise AssertionError(f"need at least 2 loss values to check a trend; got {losses!r}")
    if not all(math.isfinite(value) for value in losses):
        raise AssertionError(f"loss curve contained a non-finite value: {losses!r}")
    if losses[-1] >= losses[0]:
        raise AssertionError(f"loss did not decrease while overfitting: first={losses[0]:.4f} final={losses[-1]:.4f}")
    for previous, current in itertools.pairwise(losses):
        if current > previous + tolerance:
            raise AssertionError(
                f"loss increased beyond tolerance {tolerance}: {previous:.4f} -> {current:.4f} "
                f"(full curve {[f'{value:.4f}' for value in losses]})"
            )


def assert_checkpoint_manifest(checkpoint) -> dict:
    """Validate a checkpoint's manifest.json (AC-3).

    Accepts either a ``CheckpointRef`` or a path. A ``CheckpointRef`` carries the
    manifest in ``.metadata`` — already read back from disk by ``checkpoint_ref``
    on the box that owns the checkpoint (the only thing that works for the Modal
    backend, whose path lives on a remote volume the driver can't see). When
    given a bare path (Ray local), the manifest is read from that path instead.
    """
    manifest = getattr(checkpoint, "metadata", None)
    if not manifest:
        manifest = read_manifest(checkpoint)
    required = {"kind", "step", "base_model", "lora", "has_optimizer", "created_at"}
    missing = required - manifest.keys()
    if missing:
        raise AssertionError(f"checkpoint manifest missing keys {sorted(missing)}: {manifest!r}")
    if not isinstance(manifest["step"], int):
        raise AssertionError(f"checkpoint manifest step is not an int: {manifest['step']!r}")
    if not manifest["base_model"]:
        raise AssertionError("checkpoint manifest base_model was empty")
    if not isinstance(manifest["lora"], dict):
        raise AssertionError(f"checkpoint manifest lora is not a mapping: {manifest['lora']!r}")
    return manifest


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
            f"prompt_logprobs length {len(response.prompt_logprobs)} did not match prompt length {prompt_input.length}"
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
    parser.add_argument("--config", default="configs/overfit_smoke_test.yaml")
    parser.add_argument(
        "--backend",
        choices=["auto", "ray", "modal"],
        default="auto",
        help="Runtime backend: 'auto' uses the config's modal.enabled, "
        "'ray' forces local Ray, 'modal' forces Modal GPUs.",
    )
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args()

    config = resolve_backend_config(args.config, args.backend)
    print(f"backend={'modal' if config.modal.enabled else 'ray'}")
    service = ServiceClient(config=config)
    training = service.create_lora_training_client(user_metadata={"example": "overfit_smoke_test"})
    tokenizer = training.get_tokenizer().result()
    datum, prompt_input, target_token_count = build_sft_datum(tokenizer, args.prompt, args.target)

    baseline_sampler = service.create_sampling_client()
    baseline = sample_text(baseline_sampler, prompt_input, max_tokens=target_token_count + 8)
    print(f"before={baseline!r}")

    losses: list[float] = []
    for step in range(1, args.steps + 1):
        fb = training.forward_backward([datum], loss_fn="cross_entropy").result()
        if not math.isfinite(fb.loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {fb.loss}")
        training.optim_step(AdamParams(learning_rate=args.learning_rate, max_grad_norm=1.0)).result()
        losses.append(float(fb.loss))
        print(f"step={step:02d} loss={fb.loss:.4f}")

    assert_loss_decreases(losses)
    print(f"PASS: loss decreased from {losses[0]:.4f} to {losses[-1]:.4f} over {len(losses)} steps.")

    checkpoint = training.save_state().result()
    manifest = assert_checkpoint_manifest(checkpoint)
    print(f"PASS: checkpoint manifest valid at {checkpoint.path} (step={manifest['step']}).")

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

    download = training.save_sampler_with_download_url()
    download_command = modal_volume_get_command(service.config.modal.volume_name, download.archive_relpath)
    service.close()
    print("LoRA download:")
    print(download_command)


if __name__ == "__main__":
    main()
