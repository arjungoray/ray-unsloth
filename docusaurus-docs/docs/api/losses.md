---
sidebar_position: 6
---

# Losses

Losses are selected by the `loss_fn` string passed to `forward` or `forward_backward`.

## Cross entropy

```python
datum = Datum(
    model_input=ModelInput.from_ints(input_tokens),
    loss_fn_inputs={
        "target_tokens": target_tokens,
        "weights": weights,
    },
)

training.forward_backward([datum], loss_fn="cross_entropy").result()
```

Supported input styles:

- `target_tokens`: explicit target token for each position.
- `labels`: labels aligned with model input; the engine uses next-token positions.
- no labels/targets: uses input ids as labels.
- `weights`: optional per-token weights; `0.0` masks a token.

The engine computes logprobs only for weighted positions when possible and returns per-row `loss_fn_outputs` with token logprobs in `TensorData`.

## Importance sampling

Required fields:

- `target_tokens`
- `logprobs`
- `advantages`
- optional `weights`

```python
datum = Datum(
    model_input=ModelInput.from_ints(tokens),
    loss_fn_inputs={
        "target_tokens": target_tokens,
        "logprobs": old_policy_logprobs,
        "advantages": advantages,
        "weights": weights,
    },
)

training.forward_backward([datum], loss_fn="importance_sampling").result()
```

Token loss:

```text
- exp(current_logprob - old_logprob) * advantage * weight
```

## PPO

PPO uses the same fields as importance sampling and clips the probability ratio.

```python
training.forward_backward(
    data,
    loss_fn="ppo",
    loss_fn_config={
        "clip_low_threshold": 0.8,
        "clip_high_threshold": 1.2,
    },
).result()
```

Token loss:

```text
- min(ratio * advantage, clipped_ratio * advantage)
```

## CISPO

CISPO uses the same fields and clipping config. The clipped ratio is detached before multiplying by the current logprob.

```python
training.forward_backward(data, loss_fn="cispo").result()
```

Token loss:

```text
- detached(clipped_ratio) * advantage * current_logprob
```

## Custom losses

A custom loss receives model outputs, original datums, and a config dictionary:

```python
def my_loss(outputs, data, config):
    loss = outputs.logits.float().mean()
    return loss, {"mean_logit": float(loss.detach().cpu())}

training.forward_backward_custom(data, my_loss).result()
```

You can also register by name:

```python
training.register_custom_loss("my_loss", my_loss).result()
training.forward_backward_custom(data, "my_loss").result()
```

## Not yet implemented

These are roadmap items, not current built-ins:

- DRO
- GRPO
- DPO/preference objectives
- reward-model training
- high-level RL data assembly abstractions
