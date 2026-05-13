---
sidebar_position: 2
---

# Reinforcement Learning

RL loops use the same clients as SFT. The difference is that you sample completions, score them, build policy datums with old-policy logprobs and advantages, then train with a policy loss.

## Minimal loop shape

```python
training = service.create_lora_training_client(base_model="qwen3.5-4b")
sampler = training.create_live_sampling_client(name="policy")

for step in range(num_steps):
    rollouts = await sample_rollouts(sampler, prompts)
    datums = build_policy_datums(rollouts)

    await (await training.forward_backward_async(
        datums,
        loss_fn="importance_sampling",
    )).result_async()

    await training.optim_step_async(AdamParams(learning_rate=4e-5)).result_async()
```

## Policy datum fields

Each RL datum should include:

```python
Datum(
    model_input=ModelInput.from_ints(prompt_tokens + completion_tokens),
    loss_fn_inputs={
        "target_tokens": target_tokens,
        "logprobs": old_policy_logprobs,
        "advantages": advantages,
        "weights": weights,
    },
)
```

The examples typically set:

- prompt positions: zero weights and zero advantages,
- completion positions: weight `1.0`,
- `target_tokens`: shifted/position-aligned generated tokens,
- `logprobs`: logprobs from the sampling policy,
- `advantages`: group-relative reward advantages.

## Supported policy losses

| Loss | Use case |
| --- | --- |
| `importance_sampling` | Simple policy-gradient style update with old-policy correction. |
| `ppo` | Ratio-clipped policy update. |
| `cispo` | Clipped importance sampling policy objective variant. |

## Existing RL examples

| Example | Purpose |
| --- | --- |
| `examples/tinker_first_rl_training.py` | Small math RL tutorial shape: sample groups, grade boxed answers, compute group-relative advantages. |
| `examples/qwen3_5_9b_rl_training.py` | Qwen3.5 9B math RL shape with W&B logging. |
| `examples/qwen3_5_4b_math_dataset_rl_training.py` | Cookbook-style RL on Hugging Face math datasets. |
| `examples/qwen3_5_9b_math_dataset_rl_training.py` | Larger Qwen math-dataset RL loop. |
| `examples/qwen3_5_4b_ruler_64k_rl_training.py` | Long-context RULER RL on needle-in-a-haystack style tasks. |
| `examples/qwen3_5_4b_multitenant_rl.py` | Three concurrent RL tenants on one shared Modal A100 pool. |

## Live policy sampling

`create_live_sampling_client()` samples directly from the training actor:

```python
sampler = training.create_live_sampling_client(name="live-policy")
```

This is efficient for RL because it avoids checkpointing before each rollout. If you need separate sampler replicas, save weights and create a sampling client instead:

```python
sampler = await training.save_weights_and_get_sampling_client_async(
    name="policy-step-10",
    replicas=2,
)
```

## Degenerate groups

The RL examples skip or retry groups where all completions receive the same reward, because group-relative advantages become zero and produce no useful policy gradient. The multi-tenant and dataset examples include retry paths and metrics for this case.

## Observability

The larger RL examples log:

- reward mean/min/max,
- advantage distributions,
- policy logprob and ratio summaries,
- sample/train/prefill token counters,
- step timing,
- completion tables,
- checkpoint paths.

W&B is enabled by default in several configs. Disable it in YAML for offline smoke tests.
