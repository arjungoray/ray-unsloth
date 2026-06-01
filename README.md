<div align="center">

# ray-unsloth

**Tinker-shaped fine-tuning primitives on infrastructure you control.**

Write the loop in Python · Ray orchestrates · Unsloth trains · Modal scales

<br/>

[![Documentation](https://img.shields.io/badge/docs-online-2563eb?logo=readthedocs&logoColor=white)](https://arjungoray.github.io/ray-unsloth/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Ray](https://img.shields.io/badge/orchestration-Ray-028CF0?logo=ray&logoColor=white)](https://ray.io/)
[![Unsloth](https://img.shields.io/badge/training-Unsloth-7C3AED)](https://github.com/unslothai/unsloth)
[![Modal](https://img.shields.io/badge/GPUs-Modal-5B21B6?logo=modal&logoColor=white)](https://modal.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

</div>

<br/>

Most fine-tuning stacks hide the loop behind abstractions. **ray-unsloth** exposes it — a low-level, inspectable API modeled on the [Tinker SDK](https://arjungoray.github.io/ray-unsloth/compare-tinker) with `forward_backward`, `optim_step`, checkpoint, and sample primitives. No hosted lock-in. Drop-in `import tinker` compatibility for cookbook examples.

```text
  your Python loop
        │
        ▼
  Ray trainer / sampler actors     ← placement, concurrency, namespaces
        │
        ▼
  Unsloth + 4-bit LoRA             ← load, train, generate, save adapters
        │
        ▼
  Modal GPUs (optional)            ← remote execution, local control plane
```

<br/>

## Documentation

Full docs site: **[arjungoray.github.io/ray-unsloth](https://arjungoray.github.io/ray-unsloth/)**

- **[Quickstart](https://arjungoray.github.io/ray-unsloth/quickstart)** — install, first SFT step, smoke tests
- **[Architecture](https://arjungoray.github.io/ray-unsloth/architecture)** — Ray → Modal → Unsloth mental model
- **[Guides](https://arjungoray.github.io/ray-unsloth/guides/sft)** — SFT, RL, runtimes, checkpoints, examples
- **[API reference](https://arjungoray.github.io/ray-unsloth/api/service-client)** — clients, types, losses
- **[Configuration](https://arjungoray.github.io/ray-unsloth/configuration)** — YAML configs for L4, A100, sharding
- **[Roadmap](https://arjungoray.github.io/ray-unsloth/project/roadmap)** — what's next

<br/>

## Quickstart

```bash
pip install -e ".[dev,unsloth]"
# Modal GPUs: pip install -e ".[dev,modal]" && modal setup
```

```python
from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient

service = ServiceClient(config="configs/example.yaml")
training = service.create_lora_training_client()

tok = training.get_tokenizer().result()
ids = tok("Explain gradient accumulation.", add_special_tokens=True)["input_ids"]

training.forward_backward(
    [Datum(model_input=ModelInput.from_ints(ids), loss_fn_inputs={"labels": ids})],
    loss_fn="cross_entropy",
).result()
training.optim_step(AdamParams(learning_rate=2e-5)).result()

sampler = training.save_weights_and_get_sampling_client()
sampler.sample(
    ModelInput.from_ints(ids),
    num_samples=1,
    sampling_params=SamplingParams(max_tokens=64, temperature=0.7),
).result()
```

More examples — Tinker tutorials, RL, multi-tenant runs — in the **[quickstart guide](https://arjungoray.github.io/ray-unsloth/quickstart)**.

<br/>

## Highlights

**Tinker-compatible surface** · `ServiceClient`, `TrainingClient`, `SamplingClient`, `import tinker`

**SFT & RL primitives** · cross-entropy, importance sampling, PPO, CISPO, grouped rollouts

**Multi-tenant LoRA** · concurrent adapters on a shared GPU — SFT and RL

**Production-shaped configs** · model sharding (Qwen3.5 9B / 2× L4), 64k RULER RL, atomic checkpoints, W&B hooks

<br/>

## Development

```bash
pip install -e ".[dev]" && pytest
```

For the real-GPU integration smoke test (full SFT loop on Ray local or Modal) and how to
run it, see **[TESTING.md](TESTING.md)**.

Apache 2.0 · see [LICENSE](LICENSE)
