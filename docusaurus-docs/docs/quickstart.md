---
sidebar_position: 2
---

# Quickstart

This quickstart runs the same pattern the rest of the project uses: create a service from a runtime config, create a LoRA training client, build a `Datum`, run one backward pass, step the optimizer, and sample from the trained adapter.

## 1. Install

From the repository root:

```bash
pip install -e ".[dev,unsloth]"
```

For Modal-backed GPU runs:

```bash
pip install -e ".[dev,modal]"
modal setup
```

For dataset and Weights & Biases examples:

```bash
pip install -e ".[dev,examples]"
```

The package requires Python 3.10 or newer. Core dependencies are Ray, PyYAML, PyTorch, Transformers, and BitsAndBytes. Unsloth, Modal, and example dependencies are optional extras.

## 2. Pick a config

The default development config is:

```bash
configs/example.yaml
```

It keeps orchestration local, enables Modal for GPU work, uses the `ray-unsloth-checkpoints` Modal Volume, and defaults to the `lfm2.5-1.2b-instruct` model alias.

For a smaller Qwen run on one Modal L4:

```bash
configs/qwen3_5_4b_1x_l4.yaml
```

For concurrent tenants on one Modal A100:

```bash
configs/qwen3_5_4b_1x_a100_multitenant.yaml
configs/qwen3_5_4b_1x_a100_multitenant_rl.yaml
```

## 3. Run a minimal SFT step

Create `scratch_quickstart.py` in the repo root if you want a throwaway script:

```python
from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient

service = ServiceClient(config="configs/example.yaml")
training = service.create_lora_training_client(user_metadata={"example": "quickstart"})

tokenizer = training.get_tokenizer().result()
encoded = tokenizer("Explain gradient accumulation.", add_special_tokens=True)
tokens = encoded["input_ids"]

datum = Datum(
    model_input=ModelInput.from_ints(tokens),
    loss_fn_inputs={"labels": tokens},
)

training.forward_backward([datum], loss_fn="cross_entropy").result()
training.optim_step(AdamParams(learning_rate=2e-5, max_grad_norm=1.0)).result()

sampler = training.save_weights_and_get_sampling_client()
response = sampler.sample(
    ModelInput.from_ints(tokens),
    num_samples=1,
    sampling_params=SamplingParams(max_tokens=64, temperature=0.7),
).result()

print(response.sequences[0].text)
service.close()
```

Run it:

```bash
python scratch_quickstart.py
```

## 4. Run maintained examples

Minimal local SFT loop:

```bash
python examples/sft_loop.py
```

Overfit smoke test:

```bash
python examples/overfit_smoke_test.py --config configs/example.yaml
```

Tinker-first SFT tutorial shape:

```bash
python examples/tinker_first_sft_training.py --config configs/example.yaml
```

Tinker-first RL tutorial shape:

```bash
python examples/tinker_first_rl_training.py --config configs/example.yaml
```

Qwen3.5 4B math-dataset RL:

```bash
python examples/qwen3_5_4b_math_dataset_rl_training.py \
  --config configs/qwen3_5_4b_1x_l4.yaml \
  --dataset math \
  --dataset-limit 256
```

Qwen3.5 9B RL:

```bash
python examples/qwen3_5_9b_rl_training.py \
  --config configs/qwen3_5_9b_2x_l4_sharded.yaml
```

Multi-tenant SFT:

```bash
python examples/qwen3_5_4b_multitenant_sft.py \
  --config configs/qwen3_5_4b_1x_a100_multitenant.yaml
```

Multi-tenant RL:

```bash
python examples/qwen3_5_4b_multitenant_rl.py \
  --config configs/qwen3_5_4b_1x_a100_multitenant_rl.yaml
```

Long-context RULER RL:

```bash
python examples/qwen3_5_4b_ruler_64k_rl_training.py \
  --config configs/qwen3_5_4b_ruler_64k.yaml
```

## 5. Run tests

```bash
pytest
```

The tests are mostly lightweight unit tests and example-shape checks. They validate client facades, data types, config parsing, checkpoint manifests, sampling behavior, loss math, distributed trainer coordination, and the structure of example scripts.

## Common first-run issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `RayUnavailableError` | Ray is not installed in the current environment. | Install `pip install -e ".[dev]"` or ensure the active environment has Ray. |
| Modal import/setup error | `modal.enabled: true` but Modal is missing or unauthenticated. | Install `.[modal]` and run `modal setup`, or disable Modal in config. |
| Model load fails | Model requires a different Transformers/Unsloth combination or Hugging Face access. | Check the selected `model_configs` entry and local credentials. |
| Checkpoints not visible across Modal calls | Modal Volume has not committed/reloaded yet. | The runtime attempts best-effort commit/reload around save operations; retry after the save completes. |
| W&B error in examples | Example config enables W&B. | Install and authenticate `wandb`, or set the example's `wandb.enabled` to `false`. |
