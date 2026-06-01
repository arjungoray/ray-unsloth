# Testing

`ray-unsloth` has two test tiers:

1. **Unit tests** — fast, CPU-only, no model downloads. Run in CI on every change.
2. **Real-GPU integration smoke test** — exercises the full SFT loop (load → train →
   checkpoint → sample) on an actual GPU. Run before releases or when touching the
   model-loading, LoRA, generation, or checkpoint paths.

## Unit tests

```bash
pip install -e ".[dev]"
pytest
```

## Real-GPU integration smoke test

The integration smoke test is `examples/overfit_smoke_test.py`. It overfits a LoRA
adapter on a single canary example and asserts, end to end, that:

- the model loads via `ServiceClient` and a baseline sample runs;
- `forward_backward(cross_entropy)` + `optim_step` drive the training loss **down**;
- `save_state()` writes a checkpoint whose `manifest.json` is present and valid;
- a `SamplingClient.sample()` call completes after training; and
- greedy generation reproduces the trained canary answer (plus logprob/top-k feature checks).

The script exits non-zero (via `AssertionError`/`FloatingPointError`) if any check fails,
so it is CI-runnable on a GPU runner.

### Requirements

- A GPU (local CUDA box or a Modal account), and the matching extras installed:

  ```bash
  # Local GPU via Ray:
  pip install -e ".[dev,unsloth]"

  # Modal GPUs:
  pip install -e ".[dev,modal]" && modal setup
  ```

### Choosing the backend (one flag, no code changes)

The backend is selected by the `modal.enabled` config flag. You can either set it in the
YAML (`modal.enabled: true` → Modal, `false` → local Ray) or override it per-run with
`--backend`, which never requires editing the config:

```bash
# Use whatever the config declares (configs/overfit_smoke_test.yaml ships with modal.enabled: true):
python examples/overfit_smoke_test.py --config configs/overfit_smoke_test.yaml

# Force the local Ray backend:
python examples/overfit_smoke_test.py --config configs/overfit_smoke_test.yaml --backend ray

# Force the Modal backend:
python examples/overfit_smoke_test.py --config configs/overfit_smoke_test.yaml --backend modal
```

`--backend auto` (the default) leaves `modal.enabled` untouched.

### Why a dedicated config (`configs/overfit_smoke_test.yaml`)

The smoke test ships with its own config rather than reusing `configs/example.yaml`. The
Modal image installs `flash-attn` **only when no configured model wants vLLM** — `flash-attn`
is a torch2.8/cu12 wheel while vLLM forces torch2.10/cu130, so the two cannot coexist in one
image. `example.yaml` lists a vLLM-wanting model, which pushes the image onto the no-FA2 branch;
on an L4 the engine then falls back to an attention backend that current `transformers`
rejects. `configs/overfit_smoke_test.yaml` references only the small LFM2.5-1.2B model, so the
image takes the flash-attn branch and the engine auto-selects `flash_attention_2` on the L4
(compute capability 8.9). Point `--config` at it (the default) for the GPU run.

### Useful options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--config` | `configs/overfit_smoke_test.yaml` | Runtime config (model, LoRA, resources, backend). |
| `--backend` | `auto` | `auto` \| `ray` \| `modal` — overrides `modal.enabled`. |
| `--steps` | `16` | Number of overfitting steps. |
| `--learning-rate` | `2e-4` | Adam learning rate. |
| `--prompt` / `--target` | canary prompt/answer | The single example to overfit. |

The default config uses a small (~1.2B) model so the test fits on a single L4-class GPU.

### Expected output

A passing run prints the active backend, a decreasing per-step loss, and `PASS:` lines
for each assertion, ending with the checkpoint path and a LoRA download command:

```text
backend=ray
before='...'
step=01 loss=2.9143
...
step=16 loss=0.0421
PASS: loss decreased from 2.9143 to 0.0421 over 16 steps.
PASS: checkpoint manifest valid at /checkpoints/... (step=16).
after=' blue maple.'
PASS: generated text contains the trained canary answer.
PASS: sampling features returned ...
```
