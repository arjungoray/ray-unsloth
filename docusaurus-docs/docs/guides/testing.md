---
sidebar_position: 7
---

# Testing

Run the unit tests from the repository root:

```bash
pytest
```

## Test coverage by area

| Test file | Area |
| --- | --- |
| `tests/test_types.py` | Public dataclasses, tensor conversion, future wrappers, aliases. |
| `tests/test_clients.py` | Client facades, futures, sampler round-robin, service creation patterns. |
| `tests/test_config_checkpoints.py` | Runtime config parsing, model alias resolution, checkpoint manifests. |
| `tests/test_engine_sampling.py` | Sampling, logprobs, stop behavior, loss behavior using test doubles. |
| `tests/test_distributed_trainer.py` | DDP coordinator sharding, aggregation, optimizer alignment. |
| `tests/test_overfit_smoke_helpers.py` | Overfit smoke utility behavior. |
| `tests/test_tinker_first_sft_training.py` | First-SFT example structure and helper behavior. |
| `tests/test_tinker_first_rl_training.py` | First-RL reward, advantage, datum, and async behavior. |
| `tests/test_qwen3_5_*` | Example-specific config and helper behavior. |

## What tests currently do well

- Validate public client method shapes without requiring real GPU execution.
- Validate local config and checkpoint behavior.
- Validate token/logprob/loss edge cases with small test doubles.
- Validate examples remain wired to this repo's API.
- Validate distributed coordinator behavior without a full multi-GPU integration run.

## Gaps

The test suite does not currently provide broad real-GPU integration coverage. Important remaining test work:

- real Ray + Unsloth smoke on a GPU machine,
- Modal end-to-end integration,
- DDP integration on multiple visible GPUs,
- checkpoint save/load across Modal containers,
- long-context memory regression checks,
- compatibility tests against selected external Tinker examples.

## Development workflow

Recommended local loop:

```bash
pip install -e ".[dev]"
pytest
```

For changes in model execution paths, also run a smoke example with the relevant extra:

```bash
pip install -e ".[dev,unsloth,modal]"
python examples/overfit_smoke_test.py --config configs/example.yaml
```

For docs:

```bash
cd docusaurus-docs
npm install
npm run build
```
