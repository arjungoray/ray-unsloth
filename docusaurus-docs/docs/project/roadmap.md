---
sidebar_position: 2
---

# Roadmap

Prioritized work aligned with [Tinker API gaps](../compare-tinker.md) and runtime hardening.

## Near-term

- Fix `get_server_capabilities().features["losses"]` to advertise all implemented RL losses
- Document and test `create_live_sampling_client` and multi-replica sampler behavior
- Improve model/config mismatch error messages
- Add checkpoint retention and cleanup helpers
- Add an intentional real-GPU integration smoke path
- W&B-off defaults documented in every example config

## Tinker API parity

Training-relevant gaps from the [compatibility matrix](../compare-tinker.md):

| Priority | Item |
| --- | --- |
| High | `dro` loss |
| High | ServiceClient model introspection (`get_lora_param_count`, LR helpers) |
| Medium | RestClient `tinker://` path helpers and archive URL support |
| Medium | True async on ServiceClient factory methods |
| Medium | `get_info_async()` on TrainingClient |
| Lower | Renderer/tokenizer registry on ServiceClient |
| Lower | SamplingClient pickle / cross-process handoff |

## Training primitives

- GRPO and DPO/preference objectives
- Reward-model-driven workflows
- Stronger `loss_fn_inputs` validation per loss
- Sequence packing and microbatch controls

## Runtime & orchestration

- Multi-actor lifecycle management and health checks
- Coordinated sampler pools
- Multi-node distributed training beyond single-node DDP
- Modal image preview and lockable dependency versions

## Checkpointing

- Object-store backends
- Retention policies and lineage tracking
- Safer restore validation (model id, LoRA rank, target modules)

## Data & tokenizer ergonomics

- Chat-template and prompt/completion datum builders
- Target/weight alignment utilities
- Vision input behavior (move from placeholders to engine support)

## Observability

- Standardized engine and example metrics
- Actor health and placement metadata
- Optional structured logging hooks

## Documentation

- Lifecycle diagrams for Modal, Ray, and DDP
- Cookbook-style recipe pages for common experiment shapes
- Keep [Tinker compatibility](../compare-tinker.md) current as both projects evolve
