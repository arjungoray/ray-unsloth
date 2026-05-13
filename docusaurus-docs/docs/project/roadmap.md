---
sidebar_position: 2
---

# Roadmap

## Near-term

- Correct capability reporting so implemented policy losses are advertised accurately.
- Add explicit docs/tests for `create_live_sampling_client`.
- Improve error messages for model/config incompatibilities.
- Add a small real-GPU integration smoke path that can be run intentionally.
- Add checkpoint retention and cleanup helpers.
- Add clearer W&B-off instructions to every example config.

## Training primitives

- Add DRO.
- Add GRPO.
- Add DPO/preference objectives.
- Add reward-model-driven workflows.
- Add more robust sequence packing and microbatch controls.
- Add validation for required `loss_fn_inputs` fields per loss.

## Runtime and orchestration

- Improve multi-actor lifecycle management.
- Add coordinated sampler pools with health checks.
- Add clearer actor placement/resource diagnostics.
- Expand distributed training beyond the current single-node DDP shape.
- Make Modal package/image selection easier to inspect before launch.

## Checkpointing

- Add object-store backends.
- Add checkpoint retention policies.
- Add checkpoint lineage tracking.
- Add archive/download URL support where a backend supports it.
- Add safer restore validation for model id, LoRA rank, and target modules.

## Data and tokenizer ergonomics

- Add chat-template helpers.
- Add prompt/completion datum builders.
- Add safer target/weight alignment utilities.
- Add higher-level text convenience APIs while preserving token-level primitives.
- Clarify or implement vision input behavior.

## Observability

- Standardize metrics emitted by examples and engine primitives.
- Expose actor health and runtime placement metadata.
- Track checkpoint lineage and sampler source in metadata.
- Add optional structured logging hooks.

## Documentation

- Generate API docs from source docstrings once the API stabilizes.
- Add diagrams for Modal, Ray, and DDP lifecycles.
- Add cookbook-style recipes for SFT, RL, evaluation, and checkpoint resume.
- Keep the Tinker comparison page current as both projects evolve.
