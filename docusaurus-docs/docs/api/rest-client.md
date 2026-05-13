---
sidebar_position: 4
---

# RestClient

`RestClient` is a local filesystem-backed subset of Tinker-style checkpoint inspection APIs. It reads checkpoint `manifest.json` files below the configured `checkpoint_root`.

Create one from the service:

```python
rest = service.create_rest_client()
```

## `list_checkpoints`

```python
response = rest.list_checkpoints().result()
for checkpoint in response.checkpoints:
    print(checkpoint.path, checkpoint.step)
```

Optional filter:

```python
response = rest.list_checkpoints(training_run_id=training.session_id).result()
```

## `list_training_runs`

Groups checkpoint manifests by `session_id`:

```python
runs = rest.list_training_runs().result()
for run in runs.training_runs:
    print(run.id, len(run.checkpoints))
```

## `get_training_run`

```python
run = rest.get_training_run(training.session_id).result()
```

If checkpoints exist for that session, metadata is pulled from the first checkpoint manifest.

## `get_weights_info`

Reads a checkpoint manifest:

```python
info = rest.get_weights_info("/checkpoints/train-id/sampler-step-10").result()
print(info.metadata)
```

## `publish_checkpoint_from_tinker_path`

Marks a local manifest as published:

```python
checkpoint = rest.publish_checkpoint_from_tinker_path(path).result()
```

This is a local compatibility operation. It does not upload to hosted Tinker storage.

## Limitations

- No network API server is started.
- No pagination.
- No hosted auth.
- No remote checkpoint download URL generation.
- Discovery is based on recursive `manifest.json` scanning.
