---
sidebar_position: 6
---

# Extending ray-unsloth

The codebase is intentionally small enough to extend directly.

## Add a model alias

Add an entry under `model_configs`:

```yaml
model_configs:
  my-model:
    model:
      base_model: org/model-id
      max_seq_length: 4096
      dtype: bfloat16
      load_in_4bit: false
      fast_inference: auto
      gpu_memory_utilization: 0.85
      trust_remote_code: false
      attn_implementation: xformers
    lora:
      rank: 16
      alpha: 16
      dropout: 0.0
      target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
      random_state: 3407
      use_rslora: false
      bias: none
      use_gradient_checkpointing: unsloth
      loftq_config: null
```

Then select it:

```yaml
model:
  config: my-model
supported_models:
  - my-model
```

## Add a new built-in loss

Loss logic is concentrated in `UnslothEngine`.

1. Add validation and implementation method near `_cross_entropy_loss` and `_policy_loss`.
2. Add handling in `forward`.
3. Add handling in `forward_backward`.
4. Return per-row `loss_fn_outputs` when practical.
5. Add tests in `tests/test_engine_sampling.py` or a new focused test file.
6. Add docs under `docusaurus-docs/docs/api/losses.md`.

Keep datum fields explicit and Tinker-shaped where possible.

## Add a new runtime backend

Runtime sessions need to provide:

```python
create_training_actor(...)
create_sampler_actors(...)
close()
```

The actors or handles must expose the method names used by `TrainingClient` and `SamplingClient`.

The existing runtime boundary is:

```text
ServiceClient -> session -> actor-like handle -> TrainerActorImpl/SamplerActorImpl -> UnslothEngine
```

Follow that boundary and avoid leaking backend-specific logic into public clients.

## Add a checkpoint backend

Start in `checkpoints.py` and `RestClient`.

Current assumptions:

- paths resolve to local `Path` objects,
- manifests are JSON files,
- atomic save is directory-based,
- discovery recursively scans for `manifest.json`.

For object storage, introduce a backend abstraction rather than adding URI branches throughout the engine.

## Add a higher-level trainer

A higher-level trainer should sit outside the primitive client API. The low-level API is useful because it remains close to Tinker and easy to test.

A reasonable extension module would own:

- dataset loading,
- batching,
- checkpoint schedules,
- evaluation schedules,
- W&B or other logging,
- retry policies,
- resumability.

It should call existing `TrainingClient` and `SamplingClient` methods rather than bypassing them.

## Add better observability

Good extension points:

- enrich `ForwardOutput.metrics` and `ForwardBackwardOutput.metrics`,
- add actor health methods,
- add checkpoint lineage fields,
- standardize W&B metric names used in examples,
- expose placement/runtime metadata in `get_server_capabilities`.
