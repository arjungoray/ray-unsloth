---
sidebar_position: 8
---

# Control plane workflows

`ray-unsloth` includes local workflow primitives for running, inspecting, evaluating, and exporting training runs without giving up the low-level Tinker-shaped API.

## Smoke run with no GPU

Use the fake provider for CI and local control-plane testing:

```bash
ray-unsloth --config configs/fake_workflow.yaml doctor
ray-unsloth --config configs/fake_workflow.yaml run --steps 3 --checkpoint-name smoke
ray-unsloth --config configs/fake_workflow.yaml runs
```

The fake provider is a deterministic byte-level model. It computes real cross-entropy and applies SGD, so the workflow exercises futures, metrics, checkpoints, evals, exports, and the UI without Ray, Unsloth, or GPU downloads.

## Provider planning

Runtime providers are registered plugins. Execution providers can create sessions; planned providers render launch artifacts for clusters you attach to with `local-ray`.

```bash
ray-unsloth --config configs/fake_workflow.yaml list-providers
ray-unsloth --config configs/fake_workflow.yaml plan --provider skypilot --write-artifacts launch/
ray-unsloth --config configs/fake_workflow.yaml plan --provider kuberay --write-artifacts launch/
```

Built-ins:

| Provider | Kind | Purpose |
| --- | --- | --- |
| `local-ray` | execution | Local or attached Ray cluster. |
| `modal` | execution | Optional Modal GPU containers. |
| `fake` | execution | GPU-free in-process workflow backend. |
| `skypilot` | planned | Render a SkyPilot task for a Ray head. |
| `kuberay` | planned | Render KubeRay `RayCluster` YAML. |
| `slurm` | planned | Render an `sbatch` Ray-head script. |
| `runpod` | planned | Render BYOC marketplace attach guidance. |

## Eval and export

Create a small JSONL dataset:

```json
{"prompt": "ray", "expected": ""}
```

Run eval and export:

```bash
ray-unsloth --config configs/fake_workflow.yaml eval checkpoints/.../smoke eval.jsonl --scorer contains
ray-unsloth export checkpoints/.../smoke --target hf --output exported-smoke
```

Eval reports are stored under `<checkpoint_root>/_store/evals`. Exports support `local` and `hf` folders directly, plus `gguf`, `ollama`, `vllm`, and `sglang` plan artifacts.

## UI

Install the UI extra and serve the local control plane:

```bash
pip install -e ".[ui]"
ray-unsloth --config configs/fake_workflow.yaml serve-ui
```

The UI reads the same file-backed store as the CLI: runs, metrics, logs, checkpoints, evals, provider health, config validation, and topology plans.

## Plugin example

The sample plugin in `examples/sample_plugin` registers an eval scorer and exporter:

```yaml
plugins:
  - examples.sample_plugin
```

Plugin modules can register providers, losses, scorers, dataset loaders, exporters, loggers, deploy targets, and UI panels through `ray_unsloth.plugins`.
