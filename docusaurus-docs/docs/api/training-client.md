---
sidebar_position: 2
---

# TrainingClient

`TrainingClient` is a Tinker-shaped facade over a trainable model actor. It exposes forward, backward, optimizer, checkpoint, and sampler-creation operations.

You normally get one from `ServiceClient`:

```python
training = service.create_lora_training_client(base_model="qwen3.5-4b", rank=16)
```

## Information methods

### `get_tokenizer`

Returns a tokenizer wrapped in `FutureValueProxy`.

```python
tokenizer = training.get_tokenizer().result()
```

You can also use it like the tokenizer directly because the proxy forwards attributes and calls:

```python
tokens = training.get_tokenizer().encode("hello")
```

### `get_info`

Returns `TrainingClientInfo` with:

- `session_id`
- `base_model`
- `lora_rank`
- `step`
- metadata including max sequence length, dtype, and resolved attention implementation

## Forward and backward

### `forward`

Computes loss outputs without gradients:

```python
output = training.forward(data, loss_fn="cross_entropy").result()
print(output.loss, output.metrics)
```

### `forward_backward`

Computes loss and accumulates gradients:

```python
fb = training.forward_backward(data, loss_fn="cross_entropy").result()
print(fb.loss)
```

The optimizer is not stepped until you call `optim_step`.

### `forward_backward_async`

Returns an `AsyncMethodFuture`:

```python
future = await training.forward_backward_async(data, loss_fn="importance_sampling")
result = await future.result_async()
```

### `register_custom_loss`

Registers a custom loss callable by name on the actor:

```python
training.register_custom_loss("my_loss", my_loss_fn).result()
```

### `forward_backward_custom`

Runs a custom backward loss. The callable receives model outputs, the original `Datum` list, and `loss_fn_config`.

```python
def custom_loss(outputs, data, config):
    loss = outputs.logits.mean()
    return loss, {"logit_mean": float(loss.detach().cpu())}

training.forward_backward_custom(data, custom_loss).result()
```

## Optimizer

### `optim_step`

```python
from ray_unsloth import AdamParams

result = training.optim_step(
    AdamParams(learning_rate=4e-5, max_grad_norm=1.0)
).result()

print(result.step)
```

`AdamParams` supports:

- `learning_rate`
- `betas` or `beta1`/`beta2`
- `eps`
- `weight_decay`
- `max_grad_norm`
- `grad_clip_norm` alias

The engine creates the optimizer lazily. The selected implementation comes from `speed.optimizer`.

## Checkpointing

### `save_state`

Saves adapter and tokenizer files plus manifest:

```python
checkpoint = training.save_state(name="sft-step-10").result()
print(checkpoint.path)
```

### `save_state_with_optimizer`

Also saves `optimizer.pt`:

```python
checkpoint = training.save_state_with_optimizer(name="resume-step-10").result()
```

### `load_state`

Loads adapter state and resets optimizer state:

```python
training.load_state(checkpoint.path).result()
```

### `load_state_with_optimizer`

Loads adapter and optimizer state:

```python
training.load_state_with_optimizer(checkpoint.path).result()
```

### `save_weights_for_sampler`

Saves adapter weights intended for sampling:

```python
saved = training.save_weights_for_sampler(name="sampler-step-10").result()
sampler = training.create_sampling_client(saved.path)
```

### `save_weights_and_get_sampling_client`

Convenience method:

```python
sampler = training.save_weights_and_get_sampling_client(name="eval-step-10")
```

Behavior:

- If `replicas` is `None` or `1`, returns a sampler client pointing at the live training actor. If a path/name was provided, it saves first.
- If `replicas > 1`, saves sampler weights and asks the service to create separate sampler actors.

### `create_live_sampling_client`

`ray-unsloth` extension for RL loops:

```python
sampler = training.create_live_sampling_client(name="policy")
```

This samples directly from the training actor, avoiding a checkpoint before every rollout.

## Supported built-in losses

- `cross_entropy`
- `importance_sampling`
- `ppo`
- `cispo`

See [Losses](./losses.md) for required datum fields.
