---
sidebar_position: 5
---

# Examples

The examples are the best executable documentation for current behavior.

## Minimal examples

| File | What it covers |
| --- | --- |
| `examples/sft_loop.py` | Basic service, training client, one SFT step, sampler creation, sampling. |
| `examples/overfit_smoke_test.py` | End-to-end smoke test that trains a canary answer and validates generated output. |
| `examples/tinker_first_sft_training.py` | Tinker first-SFT tutorial shape adapted to `ray_unsloth`. |
| `examples/tinker_first_rl_training.py` | Tinker first-RL tutorial shape with grouped math rollouts. |

## Dataset RL examples

| File | What it covers |
| --- | --- |
| `examples/qwen3_5_4b_math_dataset_rl_training.py` | Qwen3.5 4B math RL on one Modal L4. |
| `examples/qwen3_5_9b_math_dataset_rl_training.py` | Qwen3.5 9B math RL with larger model settings. |
| `examples/qwen3_5_4b_ruler_64k_rl_training.py` | Long-context RL against RULER-style tasks. |

These examples use Hugging Face datasets, answer extraction/grading, group-relative advantages, optional SFT anchor loss, microbatching, and W&B logging.

## Multi-tenant examples

| File | What it covers |
| --- | --- |
| `examples/qwen3_5_4b_multitenant_sft.py` | Two concurrent SFT tenants on one shared A100 Modal pool. |
| `examples/qwen3_5_4b_multitenant_rl.py` | Three concurrent RL tenants on one shared A100 Modal pool. |

Each tenant has independent prompts, examples, LoRA adapter state, W&B run metadata, and final sampler checkpoint.

## Helper module

`examples/_tinker_helpers.py` contains reusable helper logic used by example scripts, including token and datum construction patterns.

## Example config blocks

Several YAML files include `examples:` sections. These are not consumed by `RuntimeConfig` itself, but example scripts read them directly for local experiment parameters.

Typical fields:

- `steps`
- `batch_size`
- `group_size`
- `learning_rate`
- `max_tokens`
- `temperature`
- `top_p`
- `top_k`
- `max_time`
- dataset names and splits
- W&B project/name/tags

## Running examples without W&B

Many example configs default to W&B enabled. To run without W&B, set:

```yaml
examples:
  example_name:
    wandb:
      enabled: false
```

or edit the relevant config block before running.

## How tests cover examples

The tests verify that example files:

- parse as valid Python,
- import this repo's API rather than external Tinker packages where intended,
- build policy datums correctly,
- compute boxed-answer rewards and group-relative advantages,
- skip degenerate RL groups,
- use direct async sample results,
- log expected metrics and token counters.
