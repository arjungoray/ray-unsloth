---
sidebar_position: 1
---

# Current Status

What works today, what is partial, and what to expect before production use.

## At a glance

<div class="status-grid">
  <div class="status-card">
    <strong>Core training loop</strong>
    SFT and RL primitives — forward, backward, optim, checkpoint, sample — are implemented end-to-end.
  </div>
  <div class="status-card">
    <strong>Tinker client shape</strong>
    Service, training, and sampling clients match the low-level Tinker API used by cookbook examples.
  </div>
  <div class="status-card">
    <strong>Modal + Ray</strong>
    Local orchestration with optional Modal GPU containers is production-tested in examples.
  </div>
  <div class="status-card">
    <strong>Examples</strong>
    SFT, RL, math datasets, 64k RULER, multi-tenant, and overfit smoke tests are maintained.
  </div>
</div>

## Implemented

### Clients & API

- `ServiceClient`, `TrainingClient`, `SamplingClient`, local `RestClient`
- `import tinker` compatibility alias and `tinker.types.*` shims
- Future wrappers: `RayObjectFuture`, `ImmediateFuture`, `AsyncMethodFuture`, `FutureValueProxy`

### Training & sampling

- LoRA setup via Unsloth — 4-bit load, target modules, RS-LoRA, sequence length
- Losses: `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, custom backward
- AdamW optimizer with gradient clipping (BitsAndBytes or PyTorch fallback)
- Generation with temperature, top-p/k, stop strings, seeds, prompt and generated logprobs
- Live-policy sampling, multi-replica samplers, round-robin routing

### Runtime & infra

- Ray session lifecycle, placement groups, trainer/sampler actor resources
- Modal GPU backend with dynamic image construction
- Single-node DDP coordinator for sharded training
- Multi-tenant Modal examples (concurrent LoRA on shared GPU pool)
- Attention backend auto-selection (FA2/FA3/xformers by GPU architecture)

### Checkpoints

- Atomic directory publishes with `manifest.json`
- Local paths, Modal Volumes, `local://`, and `tinker://local/...` handling
- Local manifest inspection via `RestClient`

## Partial support

| Area | Current state | What's missing |
| --- | --- | --- |
| [Tinker API parity](../compare-tinker.md) | Core train/sample/checkpoint loop works | Model introspection helpers, `dro`, RestClient archive/TTL, true ServiceClient async |
| Vision inputs | Type placeholders exist | Engine paths are text/token focused |
| RestClient | Local manifest reader | Pagination, remote archive URLs, delete/TTL |
| Distributed training | Single-node DDP | Multi-node, fault tolerance |
| Capability reporting | Returns local metadata | Under-advertises implemented RL losses |

## Known limitations

- No hosted control plane — auth, billing, remote sessions, and audit APIs are out of scope
- Model catalog is config-defined and depends on your Unsloth/Transformers stack
- Example configs assume specific Modal GPU types (L4, A100)
- Unit tests are lightweight; real-GPU integration coverage is limited
- No formal API stability guarantee yet

## Recommended posture

Treat `ray-unsloth` as an **active experimental primitive layer**. It is well-suited for research iteration, smoke tests, and controlled experiments.

Before production workloads, harden checkpoint retention, observability, dependency pinning, integration tests on target hardware, and runtime failure handling. See [Work in progress](./work-in-progress.md) for active hardening areas.

## Quick links

- [Tinker compatibility matrix](../compare-tinker.md) — implemented vs not yet
- [Roadmap](./roadmap.md) — planned work
- [Examples](../guides/examples.md) — runnable reference scripts
