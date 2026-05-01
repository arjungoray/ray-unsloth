# ray-unsloth

`ray-unsloth` is a small Python package for running Tinker-shaped low-level
fine-tuning primitives on Ray actors backed by Unsloth models.

The repository is not a full trainer or dataset framework. Your loop stays in
regular Python, while `ray-unsloth` provides client facades for training,
sampling, checkpointing, and Ray GPU placement. Unsloth owns the model loading,
LoRA adapter setup, forward/backward passes, generation, and adapter saves.

## Tinker Compatibility

The package exposes both `ray_unsloth` and a lightweight `tinker` import alias.
That means many SDK and cookbook examples can be used as-is after installing
this repository in editable mode:

```python
import tinker

service_client = tinker.ServiceClient(config="configs/example.yaml")
training_client = await service_client.create_lora_training_client_async(
    base_model="Qwen/Qwen3.5-4B",
    rank=16,
)
tokenizer = training_client.get_tokenizer()
```

Cookbook-style `ModelInput(chunks=[...])`, `EncodedTextChunk`, image chunk
placeholders, `TensorData.from_numpy(...)`, `TensorData.from_torch(...)`,
`target_tokens`/`weights` cross-entropy data, synchronous tokenizer/info access,
direct `await sampling_client.sample_async(...)`, and the common RL loss names
`importance_sampling`, `ppo`, and `cispo` are supported. The main remaining edit
for real Tinker examples is usually passing a local runtime config to
`ServiceClient` so Ray/Modal/Unsloth know where to run.

## Quickstart

Install the package for local development:

```bash
pip install -e ".[dev,unsloth]"
```

For the Modal-backed smoke test, install the Modal extra and authenticate once:

```bash
pip install -e ".[dev,modal]"
modal setup
```

Create a service from the example runtime config, run one supervised
fine-tuning step, save adapter weights, and sample from a new sampler actor:

```python
from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient

service = ServiceClient(config="configs/example.yaml")
training = service.create_lora_training_client(user_metadata={"example": "quickstart"})

tokenizer = training.get_tokenizer().result()
encoded = tokenizer("Explain gradient accumulation.", add_special_tokens=True)

datum = Datum(
    model_input=ModelInput.from_ints(encoded["input_ids"]),
    loss_fn_inputs={"labels": encoded["input_ids"]},
)

training.forward_backward([datum], loss_fn="cross_entropy").result()
training.optim_step(AdamParams(learning_rate=2e-5)).result()

sampler = training.save_weights_and_get_sampling_client()
sample = sampler.sample(
    ModelInput.from_ints(encoded["input_ids"]),
    num_samples=1,
    sampling_params=SamplingParams(max_tokens=64, temperature=0.7),
).result()
```

See `examples/sft_loop.py` for a minimal local SFT loop.

For a stricter end-to-end check, run the overfit smoke test. It trains one
canary answer for several LoRA steps, samples from the saved adapter, and fails
if generation is empty, punctuation-only, or missing the trained answer:

```bash
python examples/overfit_smoke_test.py --config configs/example.yaml
```

To run the official Tinker first-SFT tutorial as a full low-level primitive
training loop against this implementation:

```bash
python examples/tinker_first_sft_training.py --config configs/example.yaml
```

To run the companion first-RL tutorial shape, which samples grouped math
rollouts, scores boxed numerical answers, builds group-relative advantages, and
updates with the `importance_sampling` policy loss:

```bash
python examples/tinker_first_rl_training.py --config configs/example.yaml
```

The default `configs/example.yaml` keeps Ray orchestration local and sends the
Unsloth GPU work to Modal. It uses a single L4-backed Modal function, stores
adapter checkpoints in the `ray-unsloth-checkpoints` Modal Volume, and requests
zero GPUs from local Ray so it can run from a laptop:

```bash
python examples/sft_loop.py
```

## Current Features

- Tinker-style public clients: `ServiceClient`, `TrainingClient`,
  `SamplingClient`, and a small local `RestClient` for checkpoint inspection.
- Ray-backed trainer and sampler actors with configurable CPU/GPU resources,
  namespaces, placement strategy, and sampler replica count.
- Modal-backed GPU execution for resource-efficient smoke tests while keeping
  the Python training loop and Ray orchestration local.
- Unsloth model loading with LoRA configuration, 4-bit loading, dtype,
  sequence length, target modules, RS-LoRA toggle, and fast inference settings
  driven by YAML or dictionaries.
- Training primitives for `forward`, `forward_backward`, async aliases, custom
  backward losses, AdamW optimizer steps, gradient clipping, and token logprob
  computation.
- Sampling primitives for text generation, multiple return sequences, top-p,
  top-k, temperature, max token, seed, stop sequence, generated-token logprob,
  prompt logprob, and top-k prompt logprob parameters.
- Adapter checkpoint helpers with atomic directory publishing, `local://` and
  initial `tinker://local/...` path handling, and manifests for training state,
  optional optimizer state, sampler-ready weights, metadata, and local publish
  status.
- Client construction from fresh config, Tinker-style method signatures, saved
  training state, saved training state with optimizer, or exported sampler
  weights.
- Lightweight dataclass request and response types that are pickle-friendly for
  Ray, include Tinker-compatible aliases for common fields, and are easy to
  inspect in tests.
- Basic unit coverage for public client futures, sampling round-robin behavior,
  runtime config parsing, checkpoint manifests, and public data types.

## Roadmap

- Broader loss support beyond the current cross entropy, importance sampling,
  PPO, and CISPO primitives, including DRO, GRPO, DPO-style objectives, and
  reward-model-driven workflows.
- Higher-level examples for multi-step SFT, evaluation, rollout collection, and
  policy optimization while keeping the low-level primitives available.
- Stronger multi-actor orchestration, including multiple trainers, coordinated
  sampler pools, and clearer lifecycle management for Ray sessions.
- More complete checkpoint backends, such as cloud/object-store paths, checkpoint
  discovery beyond the local manifest index, and retention policies.
- Richer model and tokenizer IO, including chat-template helpers, prompt/text
  convenience APIs, and safer validation around tokenized inputs.
- Expanded observability for training and sampling metrics, actor health,
  resource placement, and checkpoint lineage.
- Integration tests that exercise real Ray and Unsloth execution on GPU
  machines in addition to the current lightweight unit tests.
- Packaging polish for optional dependencies, examples, and compatibility across
  supported Unsloth, Transformers, PyTorch, and Ray versions.

## Configuration

Runtime behavior is configured with `RuntimeConfig` or YAML. The example config
in `configs/example.yaml` defines Ray connection settings, default model and
LoRA parameters, model-specific `model_configs`, resource requests for trainer
and sampler actors, checkpoint root, and supported model metadata. Passing a
configured alias such as `base_model="gemma4-e3b-it"`,
`base_model="lfm2.5-1.2b-instruct"`, or `base_model="qwen3.5-4b"` selects
that model's full Unsloth `from_pretrained` and `get_peft_model` recipe.

## Development

Install development dependencies and run the unit tests:

```bash
pip install -e ".[dev]"
pytest
```
