"""Tiny supervised recipe gallery example."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from ray_unsloth import AdamParams, ModelInput, SamplingParams, ServiceClient
from ray_unsloth.recipes import sft_epoch, text_completion_datum

EXAMPLES = [
    {"prompt": "Complete: ray-unsloth is", "target": " a tiny training library."},
    {"prompt": "Complete: fake provider means", "target": " no GPU is required."},
    {"prompt": "Complete: recipes make", "target": " the loop reusable."},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    service = ServiceClient(config=args.config)
    training = service.create_lora_training_client(user_metadata={"example": "gallery-01-hello-sft"})
    tokenizer = training.get_tokenizer().result()
    datums = [text_completion_datum(tokenizer, row["prompt"], row["target"]) for row in EXAMPLES]
    epochs = 1 if args.smoke else 3
    losses = []

    for epoch in range(epochs):
        batch_losses = sft_epoch(
            training,
            datums,
            batch_size=len(datums),
            adam_params=AdamParams(learning_rate=0.35),
            shuffle_seed=epoch,
        )
        loss = batch_losses[0]
        losses.append(loss)
        print(f"epoch={epoch + 1} loss={loss:.4f}")

    assert all(left > right for left, right in itertools.pairwise(losses))

    checkpoint = training.save_state(name="hello-sft").result()
    prompt_tokens = tokenizer(EXAMPLES[0]["prompt"], add_special_tokens=True)["input_ids"]
    sampler = service.create_sampling_client(model_path=checkpoint.path)
    sample = sampler.sample(
        ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=12, temperature=0.0),
    ).result()
    print(f"checkpoint={checkpoint.path}")
    print(sample.sequences[0].text or "")
    service.close()


if __name__ == "__main__":
    main()
