---
sidebar_position: 3
---

# SamplingClient

`SamplingClient` is the generation and logprob facade. It can wrap one actor or many actors. When multiple actors are present, calls round-robin across replicas.

```python
sampler = service.create_sampling_client(base_model="qwen3.5-4b", replicas=2)
```

## `sample`

```python
from ray_unsloth import ModelInput, SamplingParams

response = sampler.sample(
    prompt=ModelInput.from_ints(prompt_tokens),
    num_samples=4,
    sampling_params=SamplingParams(
        max_tokens=128,
        temperature=0.8,
        top_p=0.95,
        top_k=20,
        stop=["<END>"],
        seed=123,
        logprobs_max_tokens=128,
    ),
    include_prompt_logprobs=True,
    topk_prompt_logprobs=5,
).result()
```

Returns `SampleResponse`:

```python
for sequence in response.sequences:
    print(sequence.text)
    print(sequence.tokens)
    print(sequence.logprobs)
```

## SamplingParams

| Field | Meaning |
| --- | --- |
| `max_tokens` | Maximum completion tokens. |
| `temperature` | `0` or lower means greedy decoding; positive values sample. |
| `top_p` | Nucleus sampling. |
| `top_k` | Top-k filtering when positive. |
| `stop` | Stop strings, tokenized without special tokens and trimmed from output. |
| `seed` | Torch seed for reproducible sampling. |
| `max_time` | Optional generation wall-clock cap. |
| `logprobs_max_tokens` | Cap completion logprob computation length. |

## Prompt logprobs

```python
response = sampler.sample(
    prompt=ModelInput.from_ints(prompt_tokens),
    sampling_params=SamplingParams(max_tokens=1),
    include_prompt_logprobs=True,
).result()

print(response.prompt_logprobs)
```

The first prompt token has logprob `None` because there is no previous token context.

## Top-k prompt logprobs

```python
response = sampler.sample(
    prompt=ModelInput.from_ints(prompt_tokens),
    sampling_params=SamplingParams(max_tokens=1),
    include_prompt_logprobs=True,
    topk_prompt_logprobs=5,
).result()

print(response.topk_prompt_logprobs)
```

The return shape is a list aligned to prompt positions. Each non-initial row contains `(token_id, logprob)` pairs.

## `compute_logprobs`

Computes token logprobs for a prompt:

```python
logprobs = sampler.compute_logprobs(ModelInput.from_ints(tokens)).result()
```

This is commonly used in RL loops for old-policy logprobs.

## Async methods

```python
response = await sampler.sample_async(
    prompt=ModelInput.from_ints(tokens),
    num_samples=8,
    sampling_params=SamplingParams(max_tokens=128),
)

logprobs = await sampler.compute_logprobs_async(ModelInput.from_ints(tokens))
```

The async sampling methods return resolved results directly.

## Information methods

```python
tokenizer = sampler.get_tokenizer().result()
base_model = sampler.get_base_model().result()
```

In Modal mode, `ModalActorHandle.get_tokenizer()` loads a plain local Transformers tokenizer from `model_config.base_model`. This avoids moving the full remote actor state back to the local process.
