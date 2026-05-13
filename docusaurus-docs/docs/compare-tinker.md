---
sidebar_position: 6
---

# Tinker API Comparison

This page compares `ray-unsloth` with the public Tinker API documented by Thinking Machines Lab. Tinker describes the hosted product as a cloud training API where user code controls loops while Tinker handles GPU infrastructure. The public Tinker positioning highlights four core functions: `forward_backward`, `optim_step`, `sample`, and `save_state` ([Thinking Machines Tinker page](https://www.thinkingmachines.ai/tinker/)).

`ray-unsloth` intentionally mirrors that low-level shape, but it is not the hosted Tinker service. It runs through Ray, optional Modal GPU containers, and Unsloth models.

## High-level comparison

| Area | Tinker API | ray-unsloth |
| --- | --- | --- |
| Infrastructure | Hosted remote GPU service. | Local Ray actors or Modal-backed GPU containers controlled by this repo. |
| Entry point | `tinker.ServiceClient()` reading Tinker credentials. | `ray_unsloth.ServiceClient(config=...)`; also exposes `import tinker` compatibility alias. |
| Training client | Created with `create_lora_training_client`. | Same public method name. |
| Sampling client | Created from service or saved training weights. | Same public method names; can also create live sampler from training actor. |
| Core train primitive | `forward_backward(data, loss_fn, loss_fn_config=None)` ([Tinker TrainingClient docs](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/)). | Implemented for `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, and custom backward losses. |
| Optimizer primitive | `optim_step(AdamParams(...))` using Adam-style parameters. | Implemented with BitsAndBytes AdamW variants or PyTorch AdamW fallback. |
| Sampling primitive | `sample(prompt, num_samples, sampling_params, include_prompt_logprobs=False, topk_prompt_logprobs=0)` ([Tinker SamplingClient docs](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/)). | Implemented with generation, stop trimming, seeds, prompt logprobs, top-k prompt logprobs, and generated-token logprobs. |
| Checkpoints | Tinker-managed storage and `tinker://` paths. | Filesystem/Modal Volume checkpoints with local `tinker://local/...` handling and local mapping for other `tinker://...` paths. |
| Cookbook abstractions | Official Tinker Cookbook includes configurable pipelines, evaluators, datasets, recipes, and CLI workflows. | Not included as framework abstractions; repo has examples that mimic first-SFT, first-RL, math RL, long-context RL, and multi-tenant loops. |
| Model catalog | Hosted API advertises many supported models. | Supported models are whatever configs and installed Unsloth/Transformers stack can load. Current configs include Gemma, LFM, and Qwen3.5 variants. |

## Workflow compatibility

The official Tinker quickstart describes two main workflows: SFT as `ServiceClient -> TrainingClient -> forward_backward("cross_entropy") -> optim_step -> save weights -> sample`, and RL as `TrainingClient -> on-policy sampler -> sample rollouts -> reward/logprobs -> forward_backward("importance_sampling") -> optim_step -> repeat` ([Tinker quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)).

`ray-unsloth` supports both workflow shapes:

- SFT examples: `examples/sft_loop.py`, `examples/tinker_first_sft_training.py`, and multi-tenant SFT.
- RL examples: `examples/tinker_first_rl_training.py`, Qwen math dataset RL, Qwen RULER long-context RL, Qwen 9B RL, and multi-tenant RL.

## API surface matrix

| Tinker-style symbol | ray-unsloth status | Notes |
| --- | --- | --- |
| `tinker.ServiceClient` | Supported alias | Imports from `ray_unsloth.ServiceClient`; accepts local runtime config. |
| `ServiceClient.create_lora_training_client` | Supported | Accepts `base_model`, `rank`, `seed`, target-module flags, metadata. |
| `ServiceClient.create_sampling_client` | Supported | Creates sampler actors from base model or saved model path. |
| `ServiceClient.create_training_client_from_state` | Supported | Loads adapter state without optimizer. |
| `ServiceClient.create_training_client_from_state_with_optimizer` | Supported | Loads adapter and optimizer state. |
| `ServiceClient.get_server_capabilities` | Partial | Returns local supported models and runtime feature metadata. |
| `TrainingClient.forward` | Supported | Forward loss/logprob output without gradients. |
| `TrainingClient.forward_backward` | Supported | Built-in losses listed below. |
| `TrainingClient.forward_backward_custom` | Supported with local callable contract | Callable receives model outputs, data, and config. |
| `TrainingClient.optim_step` | Supported | AdamW-style parameters plus gradient clipping. |
| `TrainingClient.save_state` | Supported | Saves adapter/tokenizer manifest without optimizer. |
| `TrainingClient.save_state_with_optimizer` | Supported | Adds `optimizer.pt`. |
| `TrainingClient.load_state` | Supported | Loads adapter state. |
| `TrainingClient.load_state_with_optimizer` | Supported | Loads adapter and optimizer state. |
| `TrainingClient.save_weights_for_sampler` | Supported | Saves sampler-ready adapter weights. |
| `TrainingClient.save_weights_and_get_sampling_client` | Supported | Returns live sampler for one replica or saved-weight sampler for multiple replicas. |
| `TrainingClient.create_live_sampling_client` | ray-unsloth extension | Samples directly from the training actor without checkpointing each rollout. |
| `SamplingClient.sample` | Supported | Returns `SampleResponse` with `GeneratedSequence` rows. |
| `SamplingClient.compute_logprobs` | Supported | Prompt token logprobs. |
| `SamplingClient.get_tokenizer` | Supported | In Modal, loads a plain local tokenizer to avoid deserializing full Unsloth actor state. |
| `SamplingClient.get_base_model` | Supported | Returns selected base model string. |
| `RestClient.list_checkpoints` | Partial local implementation | Reads local manifest files. |
| `RestClient.list_training_runs` | Partial local implementation | Groups manifests by session id. |
| `RestClient.get_weights_info` | Partial local implementation | Reads manifest metadata. |

## Type compatibility

| Type | Status | Notes |
| --- | --- | --- |
| `ModelInput` | Supported | Encoded text chunks, image placeholder chunks, `from_ints`, `to_ints`, append helpers. |
| `EncodedTextChunk` | Supported | Token sequence chunk. |
| `ImageChunk`, `ImageAssetPointerChunk` | Placeholder-compatible | Length requires `expected_tokens`; engine currently operates on tokenized text paths. |
| `TensorData` | Supported | Serializable tensor container with NumPy/Torch conversion helpers. |
| `Datum` | Supported | Holds `model_input`, `loss_fn_inputs`, and metadata. |
| `SamplingParams` | Supported subset plus local fields | `max_tokens`, `temperature`, `top_p`, `top_k`, `stop`, `seed`, `max_time`, `logprobs_max_tokens`. |
| `AdamParams` | Supported | Also accepts `beta1`, `beta2`, and `grad_clip_norm` aliases. |
| Response dataclasses | Supported local equivalents | Lightweight dataclasses, not Pydantic models. |

## Loss comparison

The Tinker docs list loss-function pages for cross-entropy, importance sampling, PPO, CISPO, DRO, and custom losses in the SDK navigation ([Tinker quickstart navigation](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)).

`ray-unsloth` currently implements:

- `cross_entropy`
- `importance_sampling`
- `ppo`
- `cispo`
- registered/custom backward losses through `forward_backward_custom`

Roadmap or incomplete:

- DRO
- GRPO
- DPO-style objectives
- reward-model-driven workflows
- higher-level cookbook loss/data abstractions

## Key behavioral differences

### Configuration is explicit

Hosted Tinker reads service credentials and provisions remote resources. `ray-unsloth` requires a runtime config that declares Ray, Modal, model, LoRA, resource, speed, checkpoint, and supported-model settings.

### Futures are local wrappers

Tinker exposes `APIFuture` objects. `ray-unsloth` exposes future-like wrappers with `.result()`, `.get()`, async result helpers, and `await` support where needed, but they are implemented around Ray object refs or immediate local values.

### Checkpoint paths are local-first

Tinker uses hosted storage and `tinker://` paths. `ray-unsloth` writes adapter directories with `manifest.json` into local filesystem paths or Modal Volumes.

### Sampler topology is explicit

Tinker abstracts sampler lifecycle behind the hosted service. `ray-unsloth` can create:

- independent sampler actor replicas from a base model or saved weights,
- a live sampling client that points at the training actor,
- sampler clients round-robinning across actor replicas.

### Cookbook compatibility is example-level

Official Tinker Cookbook APIs provide higher-level training pipelines, evaluation abstractions, recipes, storage utilities, and benchmark integrations. This repository does not reimplement those abstractions. It provides low-level primitives and examples that follow similar algorithmic shapes.

## When to use each

Use Tinker when you want the hosted service, broader model catalog, official SDK/Cookbook support, managed checkpoints, and production-grade remote training infrastructure.

Use `ray-unsloth` when you want local inspectability, Ray/Modal control, Unsloth-backed LoRA experimentation, and a Tinker-shaped surface that can be modified inside this repository.
