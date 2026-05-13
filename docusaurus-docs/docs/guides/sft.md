---
sidebar_position: 1
---

# Supervised Fine-Tuning

SFT trains the model to imitate target tokens. In `ray-unsloth`, SFT is just `cross_entropy` over `Datum` objects.

## Minimal flow

```python
from ray_unsloth import AdamParams, Datum, ModelInput, ServiceClient

service = ServiceClient(config="configs/example.yaml")
training = service.create_lora_training_client()
tokenizer = training.get_tokenizer().result()

tokens = tokenizer("The capital of France is Paris.", add_special_tokens=True)["input_ids"]
datum = Datum(
    model_input=ModelInput.from_ints(tokens),
    loss_fn_inputs={"labels": tokens},
)

training.forward_backward([datum], loss_fn="cross_entropy").result()
training.optim_step(AdamParams(learning_rate=2e-5)).result()
```

## Masking prompt tokens

For instruction data, you usually train only on the completion:

```python
prompt_tokens = tokenizer("Question: 2+2?\nAnswer:", add_special_tokens=True)["input_ids"]
answer_tokens = tokenizer(" 4", add_special_tokens=False)["input_ids"]
tokens = prompt_tokens + answer_tokens

target_tokens = tokens[1:] + [tokenizer.eos_token_id]
weights = [0.0] * len(prompt_tokens) + [1.0] * len(answer_tokens)

datum = Datum(
    model_input=ModelInput.from_ints(tokens),
    loss_fn_inputs={
        "target_tokens": target_tokens,
        "weights": weights,
    },
)
```

## Evaluation sampling

```python
sampler = training.save_weights_and_get_sampling_client(name="sft-eval")
response = sampler.sample(
    ModelInput.from_ints(prompt_tokens),
    sampling_params=SamplingParams(max_tokens=64, temperature=0.7),
).result()
```

## Existing SFT examples

| Example | Purpose |
| --- | --- |
| `examples/sft_loop.py` | Minimal one-file SFT loop. |
| `examples/tinker_first_sft_training.py` | Low-level Tinker first-SFT tutorial shape adapted to this repo. |
| `examples/overfit_smoke_test.py` | Trains a canary answer and fails if generation does not recover it. |
| `examples/qwen3_5_4b_multitenant_sft.py` | Runs two concurrent LoRA SFT tenants on a shared Modal A100 pool. |

## Practical notes

- Use `weights` to mask prompt/context tokens.
- Use `max_grad_norm` in `AdamParams` for safer small-run experiments.
- Save sampler weights by name if you want a durable path.
- Use `create_live_sampling_client` only when you are comfortable sampling from the live training actor.
