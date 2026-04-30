"""Minimal local loop using Ray Unsloth primitives.

Run on a GPU machine with Ray and Unsloth installed:

    python examples/sft_loop.py
"""

from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient


def main() -> None:
    service = ServiceClient("configs/example.yaml")
    training = service.create_lora_training_client()
    tokenizer = training.get_tokenizer().result()

    prompt = "Write a one sentence definition of gradient accumulation."
    target = " Gradient accumulation sums gradients across mini-batches before an optimizer step."
    encoded = tokenizer(prompt + target, add_special_tokens=True)
    prompt_encoded = tokenizer(prompt, add_special_tokens=True)

    labels = [-100] * len(prompt_encoded["input_ids"])
    labels.extend(encoded["input_ids"][len(prompt_encoded["input_ids"]) :])

    datum = Datum(
        model_input=ModelInput.from_ints(encoded["input_ids"]),
        loss_fn_inputs={"labels": labels},
    )

    fb = training.forward_backward([datum], loss_fn="cross_entropy").result()
    print(f"loss={fb.loss:.4f}")
    training.optim_step(AdamParams(learning_rate=2e-5)).result()

    sampler = training.save_weights_and_get_sampling_client()
    sample = sampler.sample(
        ModelInput.from_ints(prompt_encoded["input_ids"]),
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=64, temperature=0.7),
    ).result()
    print(sample.sequences[0].text)


if __name__ == "__main__":
    main()
