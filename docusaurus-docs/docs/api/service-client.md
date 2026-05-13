---
sidebar_position: 1
---

# ServiceClient

`ServiceClient` is the control-plane entry point. It loads runtime config, chooses the runtime backend, and creates training, sampling, and local checkpoint clients.

```python
from ray_unsloth import ServiceClient

service = ServiceClient(config="configs/example.yaml")
```

## Construction

```python
ServiceClient(
    user_metadata: dict[str, str] | str | RuntimeConfig | dict | None = None,
    project_id: str | None = None,
    config: str | RuntimeConfig | dict | None = None,
    base_url: str | None = None,
)
```

Notes:

- `base_url` is accepted for Tinker-shape compatibility but ignored.
- If `config` is omitted and `user_metadata` looks like a config path/object/dict, it is treated as config.
- `user_metadata`, `metadata`, and `project_id` are merged into training-run metadata.
- `modal.enabled: true` selects `ModalSession`; otherwise `RaySession`.

## Methods

### `get_server_capabilities`

Returns `GetServerCapabilitiesResponse` with supported model names, multi-trainer/sampler flags, resource limits, and runtime feature metadata.

Current feature metadata includes:

- runtime backend: `ray` or `modal`
- Ray namespace
- trainer replica count
- checkpointing support
- speed profile and optimizer settings
- advertised loss list

### `create_lora_training_client`

```python
training = service.create_lora_training_client(
    base_model="qwen3.5-4b",
    rank=16,
    seed=3407,
    user_metadata={"run": "experiment-1"},
)
```

Creates a trainable LoRA session and returns `TrainingClient`.

Important parameters:

- `base_model`: full model id or alias from `model_configs`.
- `rank`: overrides selected LoRA rank.
- `seed`: overrides selected LoRA random state.
- `train_mlp`, `train_attn`, `train_unembed`: build target modules from module-group flags.
- `metadata` and `user_metadata`: merged into engine metadata and checkpoint manifests.

### `create_sampling_client`

```python
sampler = service.create_sampling_client(
    base_model="qwen3.5-4b",
    replicas=2,
)
```

Creates sampler actor replicas for base-model sampling or saved adapter sampling.

Use `model_path` to load saved sampler weights:

```python
sampler = service.create_sampling_client(model_path="/checkpoints/train-id/sampler")
```

### `create_training_client_from_state`

Loads a trainable adapter checkpoint without optimizer state.

```python
training = service.create_training_client_from_state(
    "/checkpoints/train-id/state-step-10",
    base_model="qwen3.5-4b",
)
```

### `create_training_client_from_state_with_optimizer`

Loads adapter and optimizer state.

```python
training = service.create_training_client_from_state_with_optimizer(
    "/checkpoints/train-id/state-step-10",
    base_model="qwen3.5-4b",
)
```

### `create_rest_client`

Returns the local manifest-backed `RestClient`.

```python
rest = service.create_rest_client()
checkpoints = rest.list_checkpoints().result()
```

### `close`

Closes the runtime session. For Ray, this kills owned actors and removes placement groups. For Modal, this exits the running app context.

```python
service.close()
```

## Metadata merge order

Training metadata is merged in this order:

1. `project_id` as `"project_id"` if set.
2. service-level `user_metadata`.
3. method-level `metadata`.
4. method-level `user_metadata`.

Later values override earlier keys.
