---
sidebar_position: 3
---

# Checkpoints

Checkpoints are adapter directories with a `manifest.json`. Saves are atomic and local-first.

## Save types

| Method | Contents | Intended use |
| --- | --- | --- |
| `save_state` | adapter, tokenizer, manifest | Continue from weights without optimizer momentum. |
| `save_state_with_optimizer` | adapter, tokenizer, `optimizer.pt`, manifest | Resume training with optimizer state. |
| `save_weights_for_sampler` | adapter, tokenizer, manifest | Create sampler actors from trained weights. |

## Manifest fields

Generated manifests include:

- `kind`: `training_state` or `sampler_weights`
- `step`
- `base_model`
- `lora`
- `has_optimizer`
- `created_at`
- `session_id`
- `model_path`
- `metadata`

## Path resolution

| Input | Resolution |
| --- | --- |
| `None` | New generated path under `checkpoint_root`. |
| simple name like `step-10` | `checkpoint_root/session_id/step-10`. |
| absolute path | Used directly. |
| path-like relative value | Resolved to an absolute local path. |
| `local://...` | Prefix stripped and resolved locally. |
| `tinker://local/...` | Mapped to local absolute path. |
| other `tinker://...` | Mapped into `checkpoints/tinker/...`. |

## Atomic publishing

`atomic_checkpoint_dir` writes into a temporary sibling directory, then atomically replaces the final directory. If the save fails, the temporary directory is removed and the old checkpoint is left alone.

## Modal Volume behavior

In Modal mode, saves happen inside a Modal Volume. The runtime attempts:

- `volume.reload()` before creating an in-container actor,
- `volume.commit()` after save methods.

This is a best-effort bridge between sequential trainer and sampler calls. The underlying actor method error still surfaces if the save itself fails.

## Inspect checkpoints

```python
rest = service.create_rest_client()

all_checkpoints = rest.list_checkpoints().result()
run = rest.get_training_run(training.session_id).result()
info = rest.get_weights_info(saved.path).result()
```

## Resume training

Without optimizer:

```python
training = service.create_training_client_from_state(
    checkpoint.path,
    base_model="qwen3.5-4b",
)
```

With optimizer:

```python
training = service.create_training_client_from_state_with_optimizer(
    checkpoint.path,
    base_model="qwen3.5-4b",
)
```

## Work in progress

Current checkpointing is practical but not a full storage service. Roadmap items include:

- object-store/cloud backends,
- checkpoint discovery beyond manifest scanning,
- retention policies,
- archive URLs,
- richer lineage metadata,
- safer cross-process publish semantics.
