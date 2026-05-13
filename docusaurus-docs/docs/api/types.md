---
sidebar_position: 5
---

# Types

Public types live in `src/ray_unsloth/types.py`. They are lightweight dataclasses designed to be pickle-friendly for Ray.

## ModelInput and chunks

`ModelInput` is a sequence of chunks:

```python
from ray_unsloth import EncodedTextChunk, ModelInput

prompt = ModelInput.from_ints([1, 2, 3])
prompt = prompt.append(4)
prompt = prompt.append([5, 6])
```

Supported chunk types:

| Type | Purpose |
| --- | --- |
| `EncodedTextChunk` | Token ids. |
| `ImageChunk` | Inline image placeholder with format and expected token length. |
| `ImageAssetPointerChunk` | External image pointer placeholder. |

`ModelInput.to_ints()` only works when all chunks are encoded text. Current engine paths operate on token ids, so image chunks are compatibility placeholders rather than full vision execution support.

## TensorData

`TensorData` is a serializable tensor container:

```python
from ray_unsloth import TensorData

td = TensorData(data=[1.0, 2.0], dtype="float32", shape=[2])
arr = td.to_numpy()
tensor = td.to_torch()
```

Construct from NumPy or Torch:

```python
td = TensorData.from_numpy(np_array)
td = TensorData.from_torch(torch_tensor)
```

It normalizes float dtypes to `float32` and integer dtypes to `int64`.

## Datum

`Datum` is the atomic training/evaluation input:

```python
from ray_unsloth import Datum, ModelInput

datum = Datum(
    model_input=ModelInput.from_ints(tokens),
    loss_fn_inputs={
        "target_tokens": target_tokens,
        "weights": weights,
    },
    metadata={"source": "example"},
)
```

During initialization, nested Torch/NumPy arrays in `loss_fn_inputs` are converted to `TensorData`.

## Sampling response types

`GeneratedSequence` has:

- `tokens`
- `text`
- `logprobs`
- `finish_reason`
- `stop_reason`

`SampleResponse` has:

- `sequences`
- `prompt_logprobs`
- `topk_prompt_logprobs`

`SampleResponse` also has `.result()`, `.result_async()`, and `.get()` so resolved sample responses can be used in future-like call sites.

## Training response types

| Type | Fields |
| --- | --- |
| `ForwardOutput` | `loss`, `metrics`, `logprobs`, `loss_fn_outputs`. |
| `ForwardBackwardOutput` | `loss`, `metrics`, `loss_fn_outputs`. |
| `OptimStepResult` | `step`, `metrics`. |
| `TrainingClientInfo` | `session_id`, `base_model`, `lora_rank`, `step`, `metadata`. |

## Checkpoint response types

| Type | Purpose |
| --- | --- |
| `CheckpointRef` | Path, step, optimizer flag, manifest metadata. |
| `Checkpoint` | Local checkpoint listing entry. |
| `SaveWeightsForSamplerResponse` | Saved sampler path and checkpoint ref. |
| `TrainingRun` | Grouped checkpoint metadata by training run/session. |
| `WeightsInfoResponse` | Manifest metadata for a path. |

## Future wrappers

| Wrapper | Purpose |
| --- | --- |
| `RayObjectFuture` | Resolves Ray object refs through `ray.get`. |
| `ImmediateFuture` | Wraps local values behind `.result()`. |
| `AsyncMethodFuture` | Delays sync/async method submission. |
| `FutureValueProxy` | Acts like a resolved object while supporting `.result()`. |
