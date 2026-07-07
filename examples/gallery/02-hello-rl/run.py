"""Tiny GRPO gallery example."""

from __future__ import annotations

import argparse
from pathlib import Path

from ray_unsloth import SamplingParams, ServiceClient
from ray_unsloth.recipes import (
    GrpoConfig,
    PromptSpec,
    Rubric,
    RubricTerm,
    grpo_round,
    text_completion_datum,
)

PROMPTS = [
    "Reply with OK.",
    "Answer the prompt with OK.",
    "Say OK and nothing else.",
    "Produce the token OK.",
    "Return OK now.",
]


def contains_target(*, completion_text: str, target: str, **_: object) -> float:
    return 1.0 if target in completion_text else 0.0


def brevity(*, completion_text: str, **_: object) -> float:
    return -len(completion_text) / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    service = ServiceClient(config=args.config)
    training = service.create_lora_training_client(user_metadata={"example": "gallery-02-hello-rl"})
    tokenizer = training.get_tokenizer().result()
    prompt_bank = [PromptSpec(prompt_text=prompt, context={"target": "OK"}) for prompt in PROMPTS]
    anchor_datums = [text_completion_datum(tokenizer, prompt, " OK") for prompt in PROMPTS]
    rubric = Rubric(
        terms=[
            RubricTerm(name="contains_target", fn=contains_target, weight=1.0, z_normalize=False),
            RubricTerm(name="brevity", fn=brevity, weight=1.0, z_normalize=False),
        ]
    )
    rounds = 1 if args.smoke else 2
    config = GrpoConfig(
        group_size=4,
        prompts_per_batch=2,
        batches_per_round=2 if args.smoke else 4,
        inner_epochs=1,
        loss_fn="ppo",
        learning_rate=0.02,
        anchor_weight=0.2,
        max_tokens=6,
        temperature=1.0,
        top_p=0.95,
        seed=0,
    )

    previous_reward = None
    for round_index in range(rounds):
        report = grpo_round(training, prompt_bank, rubric, config, anchor_datums=anchor_datums)
        print(f"round={round_index + 1} mean_reward={report.mean_reward:.4f} losses={report.losses}")
        if previous_reward is not None:
            assert report.mean_reward >= previous_reward - 1e-6
        previous_reward = report.mean_reward

    checkpoint = training.save_state(name="hello-rl").result()
    sampler = service.create_sampling_client(model_path=checkpoint.path)
    sample = sampler.sample(
        tokenizer(PROMPTS[0], add_special_tokens=True)["input_ids"],
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=8, temperature=0.0),
    ).result()
    print(f"checkpoint={checkpoint.path}")
    print(sample.sequences[0].text or "")
    service.close()


if __name__ == "__main__":
    main()
