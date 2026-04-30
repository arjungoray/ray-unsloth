# ray-unsloth

`ray-unsloth` is a small Python package for running Tinker-shaped low-level
fine-tuning primitives on Ray actors backed by Unsloth models.

The repository is not a full trainer or dataset framework. Your loop stays in
regular Python, while `ray-unsloth` provides client facades for training,
sampling, checkpointing, and Ray GPU placement. Unsloth owns the model loading,
LoRA adapter setup, forward/backward passes, generation, and adapter saves.

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

service = ServiceClient("configs/example.yaml")
training = service.create_lora_training_client()

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

The default `configs/example.yaml` keeps Ray orchestration local and sends the
Unsloth GPU work to Modal. It uses a single L4-backed Modal function, stores
adapter checkpoints in the `ray-unsloth-checkpoints` Modal Volume, and requests
zero GPUs from local Ray so it can run from a laptop:

```bash
python examples/sft_loop.py
```

## Current Features

- Tinker-style public clients: `ServiceClient`, `TrainingClient`, and
  `SamplingClient`.
- Ray-backed trainer and sampler actors with configurable CPU/GPU resources,
  namespaces, placement strategy, and sampler replica count.
- Modal-backed GPU execution for resource-efficient smoke tests while keeping
  the Python training loop and Ray orchestration local.
- Unsloth model loading with LoRA configuration, 4-bit loading, dtype,
  sequence length, target modules, RS-LoRA toggle, and fast inference settings
  driven by YAML or dictionaries.
- Training primitives for `forward`, `forward_backward`, custom backward losses,
  AdamW optimizer steps, gradient clipping, and token logprob computation.
- Sampling primitives for text generation, multiple return sequences, top-p,
  top-k, temperature, max token, and seed parameters.
- Adapter checkpoint helpers with atomic directory publishing and manifests for
  training state, optional optimizer state, and sampler-ready weights.
- Client construction from fresh config, saved training state, saved training
  state with optimizer, or exported sampler weights.
- Lightweight dataclass request and response types that are pickle-friendly for
  Ray and easy to inspect in tests.
- Basic unit coverage for public client futures, sampling round-robin behavior,
  runtime config parsing, checkpoint manifests, and public data types.

## Roadmap

- Broader loss support beyond cross entropy, including first-class RL losses
  such as PPO, GRPO, DPO-style objectives, and reward-model-driven workflows.
- Higher-level examples for multi-step SFT, evaluation, rollout collection, and
  policy optimization while keeping the low-level primitives available.
- Stronger multi-actor orchestration, including multiple trainers, coordinated
  sampler pools, and clearer lifecycle management for Ray sessions.
- More complete checkpoint backends, such as cloud/object-store paths,
  resumable runs, checkpoint discovery, and retention policies.
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
in `configs/example.yaml` defines Ray connection settings, the base model,
LoRA parameters, resource requests for trainer and sampler actors, checkpoint
root, and supported model metadata.

## Development

Install development dependencies and run the unit tests:

```bash
pip install -e ".[dev]"
pytest
```