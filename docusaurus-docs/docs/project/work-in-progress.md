---
sidebar_position: 3
---

# Work In Progress

This page calls out items that are implemented enough to appear in code or examples but still need hardening.

## Multi-tenant execution

The multi-tenant SFT and RL examples are present and configurable. Remaining hardening:

- stronger isolation guarantees,
- clearer per-tenant resource accounting,
- better failure handling when one tenant fails,
- shared-pool observability,
- checkpoint retention per tenant.

## Long-context training

The RULER 64k example and config exist. Remaining hardening:

- memory regression tests,
- more explicit attention backend compatibility checks,
- better handling for models that cannot use `logits_to_keep`,
- evaluation summaries for long-context success/failure modes.

## Modal dependency images

Modal image construction is dynamic and model-dependent. Remaining hardening:

- preview command to print selected packages,
- clearer compatibility matrix,
- lockable image versions,
- faster image rebuild strategy,
- recovery docs for failed wheel resolution.

## Distributed training

Single-node DDP is implemented. Remaining hardening:

- multi-node design,
- fault handling,
- integration tests on real multi-GPU hardware,
- cleaner process-group teardown,
- sampler behavior for distributed-trained adapters.

## Tinker compatibility

The compatibility layer is useful for low-level examples. Remaining hardening:

- broader type/module alias coverage,
- behavioral parity tests for selected examples,
- clearer errors for unsupported hosted-only APIs,
- complete capability reporting,
- compatibility notes per Tinker SDK release.

## Documentation freshness

These docs were generated from the current repository state. Keep them current when:

- public client method signatures change,
- config schema changes,
- examples are added or renamed,
- runtime behavior changes,
- Tinker API compatibility changes.
