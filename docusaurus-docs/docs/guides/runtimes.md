---
sidebar_position: 4
---

# Runtimes

`ray-unsloth` can run with direct Ray actors or Modal-backed GPU actors.

## Ray runtime

Set:

```yaml
modal:
  enabled: false
```

`RaySession` initializes Ray if needed:

```python
ray.init(
    address=config.ray.address,
    namespace=config.ray.namespace,
    ignore_reinit_error=config.ray.ignore_reinit_error,
)
```

It creates:

- one trainer actor per training client,
- one or more sampler actors per sampling client,
- per-session placement groups,
- optional DDP worker actors and coordinator.

Use direct Ray when the machine or cluster already has the required GPUs and dependencies.

## Modal runtime

Set:

```yaml
modal:
  enabled: true
  gpu: L4
  volume_name: ray-unsloth-checkpoints
  volume_mount_path: /checkpoints
```

`ModalSession` still starts local Ray for orchestration, but the heavy actor method calls run inside a Modal class.

Use Modal when:

- you want to launch from a laptop,
- local Ray should request zero GPUs,
- the GPU environment should be built from config,
- checkpoints should persist in a Modal Volume.

## Modal image selection

The Modal session builds a Python image and installs model-dependent packages:

- Torch 2.8/cu12 and FlashAttention packages for flash-attention-oriented configs.
- Torch 2.10/cu13 and vLLM wheel when fast inference/vLLM is desired.
- Transformers 5.5.0 for Qwen3.5-style models.
- Transformers 4.57.6 for selected vLLM paths.
- Unsloth and Unsloth Zoo.

The package source is copied into `/root/ray_unsloth_src/ray_unsloth`.

## Multi-tenant Modal pools

The multi-tenant configs use:

```yaml
modal:
  max_inputs: 2
  trainer_pool_key: qwen3.5-4b-a100-shared
```

`trainer_pool_key` lets independent training sessions target the same warm Modal actor pool. `max_inputs` controls Modal concurrency.

Each tenant still receives its own training session id and independent LoRA adapter state.

## Single-node DDP

Enable DDP:

```yaml
distributed:
  enabled: true
  mode: ddp
  num_nodes: 1
  gpus_per_node: 2
  backend: nccl
```

Ray creates one worker per GPU and one coordinator. The coordinator:

- shards datums round-robin,
- invokes each worker,
- aggregates losses and metrics,
- restores original datum order for loss outputs,
- verifies optimizer steps remain aligned.

Current limitations:

- one node only,
- DDP only,
- no elastic membership,
- batch must contain at least one datum per rank.
