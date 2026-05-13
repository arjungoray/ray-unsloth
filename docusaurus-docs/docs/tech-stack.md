---
sidebar_position: 4
---

# Tech Stack

## Runtime dependencies

| Dependency | Role |
| --- | --- |
| Python 3.10+ | Package runtime and user training loops. |
| Ray | Local actor orchestration, placement groups, futures, and optional DDP worker placement. |
| PyTorch | Model execution, autograd, AdamW fallback, tensor handling, DDP. |
| Transformers | Tokenizer loading and generation APIs used beneath Unsloth. |
| BitsAndBytes | Optional 8-bit AdamW and paged 8-bit AdamW optimizers. |
| PyYAML | Runtime config loading. |
| Unsloth / Unsloth Zoo | Optional model loading, LoRA adapter setup, fast inference, and training/inference mode helpers. |
| Modal | Optional remote GPU execution with local control-plane orchestration. |

## Optional development dependencies

| Extra | Installs | Used for |
| --- | --- | --- |
| `dev` | `pytest` | Unit tests and example-shape tests. |
| `unsloth` | `unsloth`, `unsloth-zoo` | Real model execution. |
| `modal` | `modal` | Remote GPU execution. |
| `examples` | `datasets`, `sympy`, `wandb` | Math/RULER datasets, answer grading helpers, experiment logging. |

## Documentation stack

This docs site is a standalone Docusaurus project under `docusaurus-docs`. It uses Docusaurus 3.x with the classic preset. The package versions target the current Docusaurus v3 line; Docusaurus 3.10 was announced in April 2026 as the final v3 release preparing for v4.

Run locally:

```bash
cd docusaurus-docs
npm install
npm start
```

Build static output:

```bash
cd docusaurus-docs
npm run build
```

## Source package layout

```text
src/ray_unsloth/
  clients/
    service.py       # ServiceClient control-plane facade
    training.py      # TrainingClient facade
    sampling.py      # SamplingClient facade
    rest.py          # local checkpoint inspection facade
    _remote.py       # Ray/local call wrappers
  runtime/
    ray/             # Ray actors, session, DDP coordinator
    modal/           # Modal-backed actor handle/session
    unsloth/         # GPU-local model engine
  checkpoints.py     # atomic checkpoint helpers
  config.py          # runtime dataclasses and YAML parsing
  errors.py          # project exceptions
  types.py           # Tinker-shaped request/response dataclasses
src/tinker/
  __init__.py        # compatibility import alias
  types.py           # compatibility submodule registration
```

## Model execution stack

The engine performs model setup in this order:

1. Configure environment variables for selected speed options.
2. Resolve effective fast-inference and attention backend settings.
3. Load model/tokenizer with `FastLanguageModel.from_pretrained`.
4. Apply LoRA with `FastLanguageModel.get_peft_model`.
5. Optionally wrap the model in PyTorch `DistributedDataParallel`.
6. Load adapter and optimizer state if a checkpoint path was provided.

## Optimizer stack

`AdamParams` maps to AdamW-style parameters:

- `learning_rate`
- `betas` or `beta1`/`beta2`
- `eps`
- `weight_decay`
- `max_grad_norm` or `grad_clip_norm`

The optimizer defaults to the config's `speed.optimizer`:

- `adamw_8bit`
- `paged_adamw_8bit`
- `adamw_torch`

If BitsAndBytes is unavailable or CUDA is unavailable, the engine falls back to `torch.optim.AdamW`.
