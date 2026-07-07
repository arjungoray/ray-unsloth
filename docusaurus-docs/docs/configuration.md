---
sidebar_position: 5
---

# Configuration

Runtime behavior is controlled by `RuntimeConfig`, loaded from a YAML file, a dictionary, or an existing dataclass instance.

```python
from ray_unsloth import ServiceClient

service = ServiceClient(config="configs/example.yaml")
```

You can also pass the config as the first positional argument when it looks like a path or config object:

```python
service = ServiceClient("configs/example.yaml")
```

## Top-level schema

| Key | Dataclass | Purpose |
| --- | --- | --- |
| `ray` | `RayConfig` | Ray address, namespace, and reinit behavior. |
| `model` | `ModelConfig` or alias selector | Default model settings or `config: alias` selector. |
| `lora` | `LoRAConfig` | Default LoRA settings. |
| `model_configs` | `dict[str, ModelRuntimeConfig]` | Named model/LoRA recipes. |
| `resources` | `ResourceConfig` | Trainer and sampler CPU/GPU requests, replicas, placement strategy. |
| `speed` | `SpeedConfig` | Optimizer and performance toggles. |
| `distributed` | `DistributedConfig` | Single-node DDP configuration. |
| `modal` | `ModalConfig` | Modal app, GPU, timeout, volume, Python version. |
| `checkpoint_root` | `str` | Root path for checkpoint saves. |
| `supported_models` | `list[str]` | Capability response and advertised aliases. |
| `provider` | `str \| null` | Runtime provider name. Defaults to `local-ray`, or `modal` when legacy `modal.enabled` is true. |
| `provider_options` | `dict` | Provider-specific planning options such as `gpu`, `workers`, or `namespace`. |
| `plugins` | `list[str]` | Python modules imported at config load time; modules can call registry APIs. |
| `run_name` | `str \| null` | Optional name stored with new run records. |
| `tracking` | `bool` | Enables local run-store metrics, logs, checkpoint lineage, and eval records. |
| `tracking_root` | `str \| null` | Where the client-side run store lives. Defaults to `checkpoint_root`; set it when `checkpoint_root` is a remote volume path (e.g. Modal's `/checkpoints`) that isn't writable on the client. |

## Model selection

`RuntimeConfig.resolve_model_configs(base_model)` handles three cases:

1. `base_model is None`: use the default selected model and LoRA config.
2. `base_model` matches a key in `model_configs`: use that named recipe.
3. `base_model` matches a recipe's `model.base_model`: use that recipe.
4. Otherwise: copy the default `model` config and replace `base_model`.

This lets examples pass either friendly aliases such as `qwen3.5-4b` or full Hugging Face model ids.

## ModelConfig

| Field | Default | Meaning |
| --- | --- | --- |
| `base_model` | `unsloth/gemma-4-E2B-it` | Model id passed to Unsloth. |
| `max_seq_length` | `2048` | Context length requested from Unsloth. |
| `dtype` | `bfloat16` | Torch dtype string. |
| `load_in_4bit` | `true` | Whether to load quantized 4-bit weights. |
| `fast_inference` | `auto` | Enables Unsloth/vLLM-style fast inference when effective. |
| `gpu_memory_utilization` | `0.85` | Passed to Unsloth loader. |
| `trust_remote_code` | `true` | Hugging Face/Transformers trust flag. |
| `device_map` | `null` | Optional explicit device map. |
| `attn_implementation` | `null` | Explicit attention backend or `auto`. See [Attention backend auto-selection](#attention-backend-auto-selection). |

## LoRAConfig

| Field | Default | Meaning |
| --- | --- | --- |
| `rank` | `32` | LoRA rank. |
| `alpha` | `16` | LoRA alpha. |
| `dropout` | `0.0` | LoRA dropout. |
| `target_modules` | attention + MLP projections | Modules to adapt. |
| `bias` | `none` | Bias handling passed to Unsloth/PEFT. |
| `use_gradient_checkpointing` | `unsloth` | Gradient checkpointing mode. |
| `random_state` | `3407` | Adapter initialization seed. |
| `use_rslora` | `false` | RS-LoRA toggle. |
| `loftq_config` | `null` | Optional LoftQ config. |

`ServiceClient.create_lora_training_client` can override rank, seed, and target module groups with `rank`, `seed`, `train_mlp`, `train_attn`, and `train_unembed`.

## Resources

Resource fields are applied to Ray actor options or represented in Modal actor handles:

```yaml
resources:
  trainer_num_gpus: 0.0
  trainer_num_cpus: 0.25
  trainer_replicas: 1
  sampler_num_gpus: 0.0
  sampler_num_cpus: 0.25
  sampler_replicas: 1
  placement_strategy: PACK
```

For Modal configs, local Ray usually requests `0.0` GPUs because GPU work happens in Modal.

## SpeedConfig

```yaml
speed:
  profile: quality
  padding_free: auto
  sample_packing: auto
  optimizer: adamw_8bit
  vllm_standby: auto
  flash_attention_2: auto
  live_policy_sampling: true
```

Validation rules:

- `profile` must be `quality` or `throughput`.
- `optimizer` must be `adamw_8bit`, `paged_adamw_8bit`, or `adamw_torch`.
- `padding_free`, `sample_packing`, `vllm_standby`, and `flash_attention_2` must be `auto`, `true`, or `false`.

## Attention backend auto-selection

When `model.attn_implementation` is `auto` (or `null`) and `speed.flash_attention_2` is `auto`, the engine picks the best attention backend at load time from the detected GPU's compute capability:

| Backend | Compute capability | Example GPUs |
| --- | --- | --- |
| `flash_attention_3` | `>= 9.0` (Hopper) | H100, H200, H800, H20, GH200 (Grace Hopper) |
| `flash_attention_2` | `8.0` / `8.6` / `8.9` (Ampere + Ada Lovelace) | A100, A800, A40, A30, A16, A10, A10G, A2; RTX A6000/A5500/A5000/A4500/A4000/A3000/A2000; RTX 3050–3090 Ti; L4, L40, L40S; RTX 6000/5000/4500/4000/4000 SFF/2000 Ada; RTX 4060–4090 |
| `xformers` | all other GPUs (Volta sm_70, Turing sm_75, etc.) or missing FA kernels | T4, V100, RTX 20-series, GTX 16-series |

If the required FlashAttention package is not installed (`flash_attn` for FA2, `flash_attn_interface` for FA3), the engine transparently falls back to `xformers`. Setting `model.attn_implementation` to any explicit value (`flash_attention_2`, `flash_attention_3`, `xformers`, `sdpa`, ...) bypasses auto-selection.

## DistributedConfig

Distributed training currently supports one mode:

```yaml
distributed:
  enabled: true
  mode: ddp
  num_nodes: 1
  gpus_per_node: 2
  backend: nccl
  placement_strategy: STRICT_PACK
```

Constraints:

- If `mode` is set, distributed training is enabled.
- `mode` must be `ddp`.
- `num_nodes` must be `1`.
- `gpus_per_node` must be at least `1`.

## ModalConfig

```yaml
modal:
  enabled: true
  app_name: ray-unsloth
  gpu: L4
  timeout: 1800
  scaledown_window: 300
  max_inputs: null
  trainer_pool_key: null
  volume_name: ray-unsloth-checkpoints
  volume_mount_path: /checkpoints
  python_version: "3.11"
```

Important fields:

- `enabled`: selects `ModalSession` instead of `RaySession`.
- `gpu`: Modal GPU descriptor such as `L4`, `A100`, or `A100-80GB`.
- `volume_name` and `volume_mount_path`: persistent checkpoint bridge.
- `max_inputs`: Modal class concurrency for multi-tenant examples.
- `trainer_pool_key`: lets multiple training sessions target the same warm Modal pool.

## Provider and workflow fields

Modern configs can select providers directly:

```yaml
provider: fake
checkpoint_root: checkpoints
tracking: true
run_name: local-smoke
provider_options:
  gpu: L4
plugins:
  - examples.sample_plugin
```

Provider selection rules:

1. If `provider` is set, that provider is used.
2. If `provider` is absent and `modal.enabled: true`, the legacy Modal path selects `modal`.
3. Otherwise the default is `local-ray`.

Use `RuntimeConfig.validate()` or `ray-unsloth validate-config` to get non-throwing validation issues for provider health, DDP compatibility, and GPU-fit estimates.

## Existing configs

| Config | Purpose |
| --- | --- |
| `configs/example.yaml` | Default Modal L4 smoke/development config with model aliases. |
| `configs/qwen3_5_4b_1x_l4.yaml` | Qwen3.5 4B math RL on one Modal L4. |
| `configs/qwen3_5_9b_2x_l4_sharded.yaml` | Qwen3.5 9B examples and sharded-oriented settings. |
| `configs/qwen3_5_4b_1x_a100_multitenant.yaml` | Two concurrent Qwen3.5 4B SFT tenants on one A100 pool. |
| `configs/qwen3_5_4b_1x_a100_multitenant_rl.yaml` | Three concurrent RL tenants on one A100 pool. |
| `configs/qwen3_5_4b_ruler_64k.yaml` | Long-context RULER RL with 64k-style prompts on A100-80GB. |
| `configs/fake_workflow.yaml` | GPU-free local workflow config for CLI, UI, eval, export, and CI smoke tests. |
