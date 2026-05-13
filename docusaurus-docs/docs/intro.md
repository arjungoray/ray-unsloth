---
sidebar_position: 1
slug: /
---

# ray-unsloth

`ray-unsloth` is a Python package for running Tinker-shaped low-level fine-tuning primitives on infrastructure you control. The public API intentionally looks like the Tinker SDK: create a `ServiceClient`, create a LoRA `TrainingClient`, call `forward_backward`, call `optim_step`, save state or sampler weights, and sample from a `SamplingClient`.

The implementation is local-first:

- Ray is the orchestration layer for trainer and sampler actors.
- Modal is an optional remote GPU runtime while the user loop and Ray control plane stay local.
- Unsloth owns model loading, LoRA adapter injection, generation, forward/backward passes, optimizer execution, and adapter persistence.
- The package also exposes a lightweight `tinker` import alias so many low-level Tinker examples can be adapted with minimal changes.

## What this project is

`ray-unsloth` is a primitive layer, not a full trainer framework. It gives you the same rough control surface that researchers expect from Tinker-style APIs while keeping the implementation inspectable and runnable from this repository.

The project is strongest when you want to:

- Write the training loop yourself in ordinary Python.
- Run LoRA SFT or RL policy-gradient experiments.
- Use Ray actors to isolate model state and GPU placement.
- Use Modal as a GPU execution backend from a laptop.
- Prototype Tinker-compatible loops without depending on the hosted Tinker service.
- Keep checkpoints in local or Modal-volume-backed paths.

## What this project is not

This repository does not currently provide a complete dataset, experiment, evaluation, or hosted control-plane product. The examples include useful loops, but the core package intentionally exposes low-level operations rather than a high-level trainer abstraction.

It also does not claim full Tinker SDK coverage. Compatibility is pragmatic: the public types, clients, and common low-level primitives are implemented where they are needed by this repo's examples and tests. See [Tinker API comparison](./compare-tinker.md) for a detailed matrix.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/ray_unsloth` | Main package: clients, types, config, runtimes, checkpoints, Unsloth engine. |
| `src/tinker` | Compatibility alias and exception/type shims for Tinker-style imports. |
| `configs` | Runtime configs for local Ray, Modal L4/A100, multi-tenant runs, and long-context experiments. |
| `examples` | SFT, RL, math-dataset RL, RULER long-context RL, overfit smoke, and multi-tenant examples. |
| `tests` | Unit tests for public clients, data types, config/checkpoints, engine behavior, distributed orchestration, and examples. |
| `docusaurus-docs` | This standalone Docusaurus documentation site. |

## Core mental model

```text
user Python loop
    |
    v
ServiceClient(config)
    |
    +-- RaySession -------------------> Ray TrainerActor / SamplerActor
    |
    +-- ModalSession -> Modal function -> in-container TrainerActorImpl / SamplerActorImpl
                                      |
                                      v
                                UnslothEngine
                                      |
                                      v
                         model + tokenizer + LoRA adapter + optimizer
```

The user loop stays in control. `ray-unsloth` handles session creation, actor placement, model loading, method calls, checkpoint paths, and response types.

## Current headline capabilities

- Tinker-style `ServiceClient`, `TrainingClient`, `SamplingClient`, and local `RestClient`.
- LoRA model setup through Unsloth, configurable by YAML or dictionaries.
- SFT with `cross_entropy`.
- RL-style policy losses with `importance_sampling`, `ppo`, and `cispo`.
- Custom backward losses through `register_custom_loss` and `forward_backward_custom`.
- Sampling with `max_tokens`, temperature, top-p, top-k, seeds, stop strings, prompt logprobs, generated-token logprobs, and top-k prompt logprobs.
- Ray actor orchestration with placement groups and configurable trainer/sampler resources.
- Optional Modal GPU execution with local Ray orchestration.
- Single-node DDP coordination for sharded training experiments.
- Local and Modal-volume checkpointing with manifest files.
- Multi-tenant LoRA examples that run multiple independent training sessions against a shared Modal GPU pool.

Start with the [quickstart](./quickstart.md), then read [architecture](./architecture.md) and [API primitives](./api/service-client.md) when you need the exact control surface.
