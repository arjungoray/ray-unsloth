---
sidebar_position: 1
---

# Tinker API Compatibility

This page tracks **training-relevant** parity between [the public Tinker SDK](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/) and `ray-unsloth`.

`ray-unsloth` mirrors Tinker's low-level client shape — `ServiceClient`, `TrainingClient`, `SamplingClient`, checkpoint helpers — but runs on **Ray + Unsloth + optional Modal**, not the hosted Tinker service.

<div class="doc-callout">

**Out of scope for this matrix:** authentication, billing, project management, audit logs, remote session APIs, and hosted-only storage URLs. Those are Tinker control-plane concerns and are intentionally not part of this project.

</div>

## Status legend

| Badge | Meaning |
| --- | --- |
| <span class="api-status api-status--implemented">Implemented</span> | Works end-to-end for local/Modal runs |
| <span class="api-status api-status--partial">Partial</span> | Present but missing params, local-only behavior, or sync alias |
| <span class="api-status api-status--missing">Not yet</span> | Not implemented; may be planned |
| <span class="api-status api-status--extension">Extension</span> | ray-unsloth addition beyond Tinker |

## Summary

<div class="compatibility-summary">
  <div class="compatibility-stat"><strong>42</strong><span>Implemented</span></div>
  <div class="compatibility-stat"><strong>14</strong><span>Partial</span></div>
  <div class="compatibility-stat"><strong>18</strong><span>Not yet</span></div>
  <div class="compatibility-stat"><strong>8</strong><span>Extensions</span></div>
</div>

Core training workflows from the [Tinker quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/) are supported:

```mermaid
flowchart TB
    subgraph sft["SFT workflow"]
        direction TB
        S1["create_lora_training_client"] --> S2["forward_backward(cross_entropy)"]
        S2 --> S3["optim_step"]
        S3 --> S4["save weights → sample"]
    end

    subgraph rl["RL workflow"]
        direction TB
        R1["sample rollouts"] --> R2["reward + logprobs"]
        R2 --> R3["forward_backward(IS | PPO | CISPO)"]
        R3 --> R4["optim_step"]
        R4 --> R1
    end
```

---

## ServiceClient

Reference: [tinker.ServiceClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/)

| Method | Status | Notes |
| --- | --- | --- |
| `create_lora_training_client(...)` | <span class="api-status api-status--implemented">Implemented</span> | `base_model`, `rank`, `seed`, target-module flags, metadata |
| `create_training_client_from_state(path)` | <span class="api-status api-status--implemented">Implemented</span> | Loads adapter without optimizer |
| `create_training_client_from_state_with_optimizer(path)` | <span class="api-status api-status--implemented">Implemented</span> | Loads adapter + optimizer |
| `create_sampling_client(model_path, base_model, ...)` | <span class="api-status api-status--partial">Partial</span> | Works; `retry_config` and extra kwargs ignored |
| `create_rest_client()` | <span class="api-status api-status--implemented">Implemented</span> | Returns local manifest scanner, not HTTP server |
| `get_server_capabilities()` | <span class="api-status api-status--partial">Partial</span> | Reflects local config/models; `features["losses"]` under-reports RL losses |
| `create_lora_training_client_async(...)` | <span class="api-status api-status--partial">Partial</span> | Sync alias, not remote async RPC |
| `create_sampling_client_async(...)` | <span class="api-status api-status--partial">Partial</span> | Sync alias |
| `create_training_client_from_state_async(...)` | <span class="api-status api-status--partial">Partial</span> | Sync alias |
| `create_training_client_from_state_with_optimizer_async(...)` | <span class="api-status api-status--partial">Partial</span> | Sync alias |
| `get_server_capabilities_async()` | <span class="api-status api-status--partial">Partial</span> | Sync alias |
| `get_registered_renderer_names()` | <span class="api-status api-status--missing">Not yet</span> | Tinker renderer registry |
| `get_renderer(name)` | <span class="api-status api-status--missing">Not yet</span> | |
| `get_registered_tokenizer_names()` | <span class="api-status api-status--missing">Not yet</span> | |
| `get_tokenizer(name)` | <span class="api-status api-status--missing">Not yet</span> | Tokenizer access is on Training/Sampling clients |
| `get_recommended_renderer_name(s)` | <span class="api-status api-status--missing">Not yet</span> | |
| `get_model_attributes(model)` | <span class="api-status api-status--missing">Not yet</span> | |
| `get_lora_param_count(...)` | <span class="api-status api-status--missing">Not yet</span> | Model introspection helpers |
| `get_lora_lr_multiplier(...)` | <span class="api-status api-status--missing">Not yet</span> | |
| `get_lora_lr_over_full_finetune_lr(...)` | <span class="api-status api-status--missing">Not yet</span> | |
| `get_full_finetune_param_count(...)` | <span class="api-status api-status--missing">Not yet</span> | |
| `get_full_finetune_lr_multiplier(...)` | <span class="api-status api-status--missing">Not yet</span> | |
| `get_lr(...)` | <span class="api-status api-status--missing">Not yet</span> | |
| `get_last_checkpoint(...)` | <span class="api-status api-status--missing">Not yet</span> | |
| `close()` | <span class="api-status api-status--extension">Extension</span> | Tear down Ray/Modal session |
| `attach_sampler_download_url(response)` | <span class="api-status api-status--extension">Extension</span> | Modal/local LoRA download URLs |

**Entry-point difference:** Tinker uses `ServiceClient()` with hosted credentials. `ray-unsloth` requires `ServiceClient(config="configs/example.yaml")` (YAML, dict, or `RuntimeConfig`).

---

## TrainingClient

Reference: [tinker.TrainingClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/)

| Method | Status | Notes |
| --- | --- | --- |
| `get_info()` | <span class="api-status api-status--implemented">Implemented</span> | Returns `TrainingClientInfo` |
| `get_tokenizer()` | <span class="api-status api-status--implemented">Implemented</span> | Future-like `.result()` wrapper |
| `forward(data, loss_fn)` | <span class="api-status api-status--implemented">Implemented</span> | No-grad forward |
| `forward_backward(data, loss_fn, loss_fn_config)` | <span class="api-status api-status--implemented">Implemented</span> | Built-in + registered custom losses |
| `forward_backward_custom(data, loss_fn, loss_fn_config)` | <span class="api-status api-status--implemented">Implemented</span> | Callable or registered name |
| `optim_step(AdamParams(...))` | <span class="api-status api-status--implemented">Implemented</span> | AdamW + grad clipping |
| `save_state(path)` | <span class="api-status api-status--implemented">Implemented</span> | Adapter checkpoint; `ttl_seconds` ignored |
| `load_state(path)` | <span class="api-status api-status--implemented">Implemented</span> | |
| `load_state_with_optimizer(path)` | <span class="api-status api-status--implemented">Implemented</span> | |
| `save_weights_for_sampler(path)` | <span class="api-status api-status--implemented">Implemented</span> | Sampler-ready adapter weights |
| `save_weights_and_get_sampling_client(...)` | <span class="api-status api-status--implemented">Implemented</span> | Live actor for 1 replica; saves first for N>1 |
| `create_sampling_client(model_path)` | <span class="api-status api-status--implemented">Implemented</span> | Delegates to `ServiceClient` |
| `forward_async(...)` | <span class="api-status api-status--implemented">Implemented</span> | True async submission |
| `forward_backward_async(...)` | <span class="api-status api-status--implemented">Implemented</span> | |
| `forward_backward_custom_async(...)` | <span class="api-status api-status--implemented">Implemented</span> | |
| `optim_step_async(...)` | <span class="api-status api-status--implemented">Implemented</span> | |
| `save_state_async(...)` | <span class="api-status api-status--implemented">Implemented</span> | |
| `load_state_async(...)` | <span class="api-status api-status--implemented">Implemented</span> | |
| `load_state_with_optimizer_async(...)` | <span class="api-status api-status--implemented">Implemented</span> | |
| `save_weights_for_sampler_async(...)` | <span class="api-status api-status--implemented">Implemented</span> | |
| `save_weights_and_get_sampling_client_async(...)` | <span class="api-status api-status--implemented">Implemented</span> | True async |
| `get_info_async()` | <span class="api-status api-status--missing">Not yet</span> | |
| `create_sampling_client_async(...)` | <span class="api-status api-status--partial">Partial</span> | Sync alias |
| `register_custom_loss(name, fn)` | <span class="api-status api-status--extension">Extension</span> | Named custom loss registry |
| `save_state_with_optimizer(...)` | <span class="api-status api-status--extension">Extension</span> | Explicit optimizer checkpoint method |
| `save_sampler_with_download_url(...)` | <span class="api-status api-status--extension">Extension</span> | Signed LoRA tarball export |
| `create_live_sampling_client(...)` | <span class="api-status api-status--extension">Extension</span> | On-policy sampling without checkpoint round-trip |
| `compute_logprobs(prompt)` | <span class="api-status api-status--extension">Extension</span> | On training actor (Tinker exposes on SamplingClient) |

---

## SamplingClient

Reference: [tinker.SamplingClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/)

| Method | Status | Notes |
| --- | --- | --- |
| `sample(prompt, num_samples, sampling_params, ...)` | <span class="api-status api-status--implemented">Implemented</span> | Stop trimming, seeds, prompt/generated logprobs |
| `compute_logprobs(prompt)` | <span class="api-status api-status--implemented">Implemented</span> | Prompt token logprobs |
| `get_tokenizer()` | <span class="api-status api-status--implemented">Implemented</span> | Modal loads tokenizer locally to avoid actor deserialization |
| `get_base_model()` | <span class="api-status api-status--implemented">Implemented</span> | |
| `sample_async(...)` | <span class="api-status api-status--implemented">Implemented</span> | Returns resolved response |
| `compute_logprobs_async(...)` | <span class="api-status api-status--implemented">Implemented</span> | |
| `get_base_model_async()` | <span class="api-status api-status--partial">Partial</span> | Wraps sync call |
| `SamplingClient.create(...)` | <span class="api-status api-status--missing">Not yet</span> | Classmethod factory |
| `on_queue_state_change(callback)` | <span class="api-status api-status--missing">Not yet</span> | Hosted queue hooks |
| Pickle / cross-process handoff | <span class="api-status api-status--missing">Not yet</span> | Tinker clients are picklable for multiprocessing |

**Topology:** `ray-unsloth` supports independent sampler replicas (round-robin), saved-weight samplers, and live-policy sampling from the training actor.

---

## RestClient (checkpoint inspection)

Reference: [tinker.RestClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/restclient/)

Only checkpoint and weight-inspection methods relevant to local training are listed. Hosted control-plane endpoints are omitted.

| Method | Status | Notes |
| --- | --- | --- |
| `list_training_runs()` | <span class="api-status api-status--partial">Partial</span> | Groups local manifests by session id |
| `get_training_run(training_run_id)` | <span class="api-status api-status--partial">Partial</span> | Local filesystem only |
| `list_checkpoints(training_run_id)` | <span class="api-status api-status--partial">Partial</span> | Scans `checkpoint_root`; no cursor pagination |
| `get_weights_info(path)` | <span class="api-status api-status--partial">Partial</span> | Reads local manifest |
| `publish_checkpoint_from_tinker_path(path)` | <span class="api-status api-status--partial">Partial</span> | Sets `published=True` in local manifest only |
| `get_training_run_by_tinker_path(path)` | <span class="api-status api-status--missing">Not yet</span> | |
| `get_weights_info_by_tinker_path(path)` | <span class="api-status api-status--missing">Not yet</span> | |
| `get_checkpoint_archive_url(...)` | <span class="api-status api-status--missing">Not yet</span> | Hosted archive URLs |
| `get_checkpoint_archive_url_from_tinker_path(...)` | <span class="api-status api-status--missing">Not yet</span> | |
| `delete_checkpoint(...)` | <span class="api-status api-status--missing">Not yet</span> | |
| `delete_checkpoint_from_tinker_path(...)` | <span class="api-status api-status--missing">Not yet</span> | |
| `unpublish_checkpoint_from_tinker_path(...)` | <span class="api-status api-status--missing">Not yet</span> | |
| `set_checkpoint_ttl_from_tinker_path(...)` | <span class="api-status api-status--missing">Not yet</span> | TTL accepted on save but not enforced |
| `list_user_checkpoints(...)` | <span class="api-status api-status--missing">Not yet</span> | |
| `get_sampler(sampler_id)` | <span class="api-status api-status--missing">Not yet</span> | |

---

## Loss functions

Reference: [Tinker losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/)

| Loss | Status | `loss_fn_inputs` | Notes |
| --- | --- | --- | --- |
| `cross_entropy` | <span class="api-status api-status--implemented">Implemented</span> | `target_tokens`/`labels`, optional `weights` | SFT |
| `importance_sampling` | <span class="api-status api-status--implemented">Implemented</span> | `target_tokens`, `logprobs`, `advantages`, optional `weights` | RL |
| `ppo` | <span class="api-status api-status--implemented">Implemented</span> | Same as IS | `clip_low_threshold`, `clip_high_threshold` in config |
| `cispo` | <span class="api-status api-status--implemented">Implemented</span> | Same as IS | Clip thresholds in config |
| `custom` (callable) | <span class="api-status api-status--implemented">Implemented</span> | User-defined | Via `forward_backward_custom` |
| `dro` | <span class="api-status api-status--missing">Not yet</span> | — | Listed in Tinker SDK |
| GRPO | <span class="api-status api-status--missing">Not yet</span> | — | Roadmap |
| DPO / preference objectives | <span class="api-status api-status--missing">Not yet</span> | — | Roadmap |
| Reward-model-driven losses | <span class="api-status api-status--missing">Not yet</span> | — | Roadmap |

---

## Data types

| Type | Status | Notes |
| --- | --- | --- |
| `ModelInput` | <span class="api-status api-status--implemented">Implemented</span> | Chunks, `from_ints`, `to_ints`, append |
| `EncodedTextChunk` | <span class="api-status api-status--implemented">Implemented</span> | |
| `Datum` | <span class="api-status api-status--implemented">Implemented</span> | Auto-converts nested tensors |
| `TensorData` | <span class="api-status api-status--implemented">Implemented</span> | NumPy/Torch conversion |
| `SamplingParams` | <span class="api-status api-status--implemented">Implemented</span> | `max_tokens`, temperature, top-p/k, stop, seed, logprob limits |
| `AdamParams` | <span class="api-status api-status--implemented">Implemented</span> | `beta1`/`beta2`/`grad_clip_norm` aliases |
| `SampleResponse` / `GeneratedSequence` | <span class="api-status api-status--implemented">Implemented</span> | Text, tokens, logprobs, finish reason |
| `ForwardBackwardOutput` / `OptimStepResult` | <span class="api-status api-status--implemented">Implemented</span> | |
| `Checkpoint` / `TrainingRun` / capabilities types | <span class="api-status api-status--implemented">Implemented</span> | Local dataclass equivalents |
| `ImageChunk` / `ImageAssetPointerChunk` | <span class="api-status api-status--partial">Partial</span> | Type-compatible placeholders; engine is text/token focused |
| HTTP request types (`ForwardRequest`, etc.) | <span class="api-status api-status--missing">Not yet</span> | Not needed for in-process client API |
| `Cursor` pagination type | <span class="api-status api-status--missing">Not yet</span> | Type exists; RestClient doesn't paginate |
| `SamplerDownloadResponse` | <span class="api-status api-status--extension">Extension</span> | Modal download metadata |

Types use **dataclasses** (pickle/Ray friendly) rather than Pydantic models.

---

## Compatibility alias

| Symbol | Status | Notes |
| --- | --- | --- |
| `import tinker` | <span class="api-status api-status--implemented">Implemented</span> | Re-exports `ray_unsloth` clients and types |
| `tinker.types.*` submodule shims | <span class="api-status api-status--implemented">Implemented</span> | Cookbook import paths |
| Exception shims (`TinkerError`, etc.) | <span class="api-status api-status--implemented">Implemented</span> | |

---

## Behavioral differences

### Configuration

Hosted Tinker provisions remote GPUs from credentials. `ray-unsloth` reads a **runtime config** (Ray resources, Modal image, model aliases, checkpoint root, LoRA defaults).

### Futures

Tinker returns hosted `APIFuture` objects. `ray-unsloth` wraps Ray object refs or immediate values in local future proxies with `.result()`, `.get()`, and selective async support.

### Checkpoints

Tinker uses hosted `tinker://` storage. `ray-unsloth` writes atomic adapter directories with `manifest.json` to local paths or Modal Volumes. `tinker://local/...` and mapped `tinker://...` paths are supported for compatibility.

### Cookbook abstractions

The [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) provides pipelines, evaluators, dataset loaders, and CLI recipes. **This repo does not reimplement those.** It provides low-level primitives and examples that follow similar algorithmic shapes (`tinker_first_sft_training`, `tinker_first_rl_training`, math RL, RULER, multi-tenant).

---

## When to use which

| Use **Tinker** when… | Use **ray-unsloth** when… |
| --- | --- |
| You want the hosted GPU service | You want Ray/Modal/Unsloth control |
| You need the full model catalog and renderer registry | You configure models via YAML + Unsloth |
| You want managed `tinker://` checkpoint storage | You want local or Modal Volume checkpoints |
| You rely on Cookbook pipelines out of the box | You write the loop and use examples as templates |

---

## Planned parity work

See [Roadmap](./project/roadmap.md) for details. Highest-impact gaps:

1. **`dro` loss** and broader RL objective coverage
2. **ServiceClient model introspection** helpers (`get_lora_param_count`, renderer registry)
3. **RestClient** archive URLs, delete/TTL, and `tinker://` path helpers
4. **Capability reporting** — advertise implemented RL losses in `get_server_capabilities()`
5. **True async** on ServiceClient factory methods
6. **Vision/multimodal** — move image chunks from placeholders to engine support
