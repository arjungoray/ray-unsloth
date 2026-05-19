# ray-unsloth

**Tinker-shaped fine-tuning primitives on infrastructure you control.**

Write the training loop in plain Python. Ray orchestrates GPU actors. Unsloth runs the model. Modal optional — train from a laptop, scale on cloud GPUs.

[![Documentation](https://img.shields.io/badge/docs-ray--unsloth-2563eb?style=for-the-badge&logo=readthedocs&logoColor=white)](https://arjungoray.github.io/ray-unsloth/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=for-the-badge)](LICENSE)

---

## Why ray-unsloth?

Most fine-tuning stacks hide the loop behind abstractions. **ray-unsloth** does the opposite: a low-level, inspectable API modeled on the [Tinker SDK](https://arjungoray.github.io/ray-unsloth/compare-tinker) — `forward_backward`, `optim_step`, checkpoint, sample — with no hosted lock-in.

| | |
|---|---|
| **Your code** | SFT, RL, math rollouts, multi-tenant LoRA — you own the loop |
| **Ray** | Trainer & sampler actors, placement, concurrency, namespaces |
| **Unsloth** | 4-bit LoRA load, forward/backward, generation, adapter saves |
| **Modal** | Optional remote GPU backend; Ray stays local on your machine |

Drop-in `import tinker` compatibility means many Tinker cookbook examples run with minimal edits.

---

## Documentation

**Full docs:** [**arjungoray.github.io/ray-unsloth**](https://arjungoray.github.io/ray-unsloth/)

| | |
|---|---|
| [Quickstart](https://arjungoray.github.io/ray-unsloth/quickstart) | Install, first SFT step, smoke tests |
| [Architecture](https://arjungoray.github.io/ray-unsloth/architecture) | Ray → Modal → Unsloth mental model |
| [Guides](https://arjungoray.github.io/ray-unsloth/guides/sft) | SFT, RL, runtimes, checkpoints, examples |
| [API reference](https://arjungoray.github.io/ray-unsloth/api/service-client) | `ServiceClient`, `TrainingClient`, `SamplingClient` |
| [Configuration](https://arjungoray.github.io/ray-unsloth/configuration) | YAML runtime configs for L4, A100, sharding |
| [Roadmap](https://arjungoray.github.io/ray-unsloth/project/roadmap) | What's next |

---

## Quickstart

```bash
pip install -e ".[dev,unsloth]"
```

```python
from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient

service = ServiceClient(config="configs/example.yaml")
training = service.create_lora_training_client()

tokenizer = training.get_tokenizer().result()
encoded = tokenizer("Explain gradient accumulation.", add_special_tokens=True)

training.forward_backward(
    [Datum(
        model_input=ModelInput.from_ints(encoded["input_ids"]),
        loss_fn_inputs={"labels": encoded["input_ids"]},
    )],
    loss_fn="cross_entropy",
).result()
training.optim_step(AdamParams(learning_rate=2e-5)).result()

sampler = training.save_weights_and_get_sampling_client()
sampler.sample(
    ModelInput.from_ints(encoded["input_ids"]),
    num_samples=1,
    sampling_params=SamplingParams(max_tokens=64, temperature=0.7),
).result()
```

For Modal-backed GPU runs: `pip install -e ".[dev,modal]"` then `modal setup`. See the [quickstart guide](https://arjungoray.github.io/ray-unsloth/quickstart) for smoke tests, Tinker tutorials, RL, and multi-tenant examples.

---

## Highlights

- **Tinker-compatible clients** — `ServiceClient`, `TrainingClient`, `SamplingClient`, plus a `tinker` import alias
- **LoRA SFT & RL** — cross-entropy, importance sampling, PPO, CISPO; grouped rollouts and policy updates
- **Multi-tenant training** — concurrent LoRA sessions on a shared GPU (SFT and RL)
- **Model sharding** — e.g. Qwen3.5 9B across two L4s on Modal
- **Long-context RL** — RULER-style 64k experiments
- **Checkpointing** — atomic adapter publishes, training state, sampler-ready weights
- **YAML-driven runtime** — Ray resources, Modal functions, model recipes, W&B hooks

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
