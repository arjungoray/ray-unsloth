---
sidebar_position: 2
---

# Quickstart

Get from zero to a working SFT step, then branch into the workflow you care about.

<div class="doc-callout doc-callout--tip">

**Prerequisites:** Python 3.10+, Git, and a machine with enough disk for model weights. GPU work can run locally or via [Modal](./guides/runtimes.md).

</div>

<ol class="step-list">

<li>
<strong>Install</strong>

From the repository root:

```bash
pip install -e ".[dev,unsloth]"
```

For Modal-backed GPU runs:

```bash
pip install -e ".[dev,modal]"
modal setup
```

For dataset and W&B examples:

```bash
pip install -e ".[dev,examples]"
```

</li>

<li>
<strong>Pick a config</strong>

| Config | Use when |
| --- | --- |
| [`configs/example.yaml`](https://github.com/arjungoray/ray-unsloth/blob/main/configs/example.yaml) | Default dev setup — local Ray + Modal L4 |
| [`configs/qwen3_5_4b_1x_l4.yaml`](https://github.com/arjungoray/ray-unsloth/blob/main/configs/qwen3_5_4b_1x_l4.yaml) | Qwen3.5 4B on one L4 |
| [`configs/qwen3_5_4b_1x_a100_multitenant.yaml`](https://github.com/arjungoray/ray-unsloth/blob/main/configs/qwen3_5_4b_1x_a100_multitenant.yaml) | Multi-tenant SFT on one A100 |

See [Configuration](./configuration.md) for the full schema.

</li>

<li>
<strong>Run one SFT step</strong>

```python
from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient

service = ServiceClient(config="configs/example.yaml")
training = service.create_lora_training_client(user_metadata={"example": "quickstart"})

tokenizer = training.get_tokenizer().result()
encoded = tokenizer("Explain gradient accumulation.", add_special_tokens=True)
tokens = encoded["input_ids"]

training.forward_backward(
    [Datum(model_input=ModelInput.from_ints(tokens), loss_fn_inputs={"labels": tokens})],
    loss_fn="cross_entropy",
).result()
training.optim_step(AdamParams(learning_rate=2e-5, max_grad_norm=1.0)).result()

sampler = training.save_weights_and_get_sampling_client()
print(sampler.sample(
    ModelInput.from_ints(tokens),
    num_samples=1,
    sampling_params=SamplingParams(max_tokens=64, temperature=0.7),
).result().sequences[0].text)

service.close()
```

Save as `scratch_quickstart.py` and run `python scratch_quickstart.py`.

</li>

<li>
<strong>Validate with maintained examples</strong>

```bash
python examples/sft_loop.py
python examples/overfit_smoke_test.py --config configs/example.yaml
pytest
```

The overfit smoke test trains a canary answer and fails if generation is empty or wrong — a good end-to-end sanity check.

</li>

</ol>

## Choose your workflow

<div class="path-grid">

<div class="path-card">

### Supervised fine-tuning

Tinker-first tutorial shape:

```bash
python examples/tinker_first_sft_training.py --config configs/example.yaml
```

→ [SFT guide](./guides/sft.md)

</div>

<div class="path-card">

### Reinforcement learning

Grouped math rollouts + policy update:

```bash
python examples/tinker_first_rl_training.py --config configs/example.yaml
```

→ [RL guide](./guides/rl.md)

</div>

<div class="path-card">

### Larger models & datasets

Qwen 9B sharded RL, math datasets, 64k RULER:

→ [Examples catalog](./guides/examples.md)

</div>

<div class="path-card">

### Multi-tenant LoRA

Concurrent adapters on one GPU:

```bash
python examples/qwen3_5_4b_multitenant_sft.py \
  --config configs/qwen3_5_4b_1x_a100_multitenant.yaml
```

→ [Runtimes guide](./guides/runtimes.md)

</div>

</div>

## Tinker cookbook port

Many low-level Tinker examples work with minimal edits:

```python
import tinker

service = tinker.ServiceClient(config="configs/example.yaml")
training = await service.create_lora_training_client_async(
    base_model="Qwen/Qwen3.5-4B",
    rank=16,
)
```

The main change is passing a local runtime `config` instead of hosted credentials. See [Tinker API compatibility](./compare-tinker.md) for the full parity matrix.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `RayUnavailableError` | Ray not installed | `pip install -e ".[dev]"` |
| Modal import/setup error | Modal missing or unauthenticated | Install `.[modal]`, run `modal setup`, or disable Modal in config |
| Model load fails | HF access or version mismatch | Check `model_configs` entry and credentials |
| Checkpoints not visible on Modal | Volume not committed yet | Retry after save completes; see [Checkpoints](./guides/checkpoints.md) |
| W&B error in examples | W&B enabled in config | `pip install wandb && wandb login`, or set `wandb.enabled: false` |

## Next steps

1. Read [Architecture](./architecture.md) to understand Ray vs Modal execution
2. Browse the [API reference](./api/service-client.md)
3. Check [Current status](./project/current-status.md) before production use
