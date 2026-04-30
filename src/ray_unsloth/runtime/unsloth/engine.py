"""GPU-local Unsloth model operations."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ray_unsloth.checkpoints import (
    atomic_checkpoint_dir,
    base_manifest,
    checkpoint_ref,
    new_checkpoint_path,
    read_manifest,
    resolve_path,
    write_manifest,
)
from ray_unsloth.config import LoRAConfig, ModelConfig
from ray_unsloth.errors import UnsupportedLossError
from ray_unsloth.types import (
    AdamParams,
    CheckpointRef,
    CustomLoss,
    Datum,
    ForwardBackwardOutput,
    ForwardOutput,
    GeneratedSequence,
    ModelInput,
    OptimStepResult,
    SampleResponse,
    SamplingParams,
    SaveWeightsForSamplerResponse,
    TensorData,
    TrainingClientInfo,
)


def _torch_dtype(name: str):
    import torch

    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(name.lower(), None)


class UnslothEngine:
    """Owns one Unsloth model, tokenizer, optimizer, and gradient state."""

    def __init__(
        self,
        *,
        session_id: str,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        checkpoint_root: str,
        model_path: str | None = None,
        with_optimizer: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.model_config = model_config
        self.lora_config = lora_config
        self.checkpoint_root = checkpoint_root
        self.metadata = dict(metadata or {})
        self.step = 0
        self.optimizer = None
        self._custom_losses: dict[str, CustomLoss] = {}

        self.model, self.tokenizer = self._load_model(model_path=model_path)
        if model_path:
            self.load_state(model_path, with_optimizer=with_optimizer)

    def _load_model(self, model_path: str | None = None):
        from unsloth import FastLanguageModel

        model_name = self.model_config.base_model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.model_config.max_seq_length,
            dtype=_torch_dtype(self.model_config.dtype),
            load_in_4bit=self.model_config.load_in_4bit,
            fast_inference=self.model_config.fast_inference,
            gpu_memory_utilization=self.model_config.gpu_memory_utilization,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_config.rank,
            target_modules=self.lora_config.target_modules,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.lora_config.random_state,
            use_rslora=self.lora_config.use_rslora,
            loftq_config=None,
        )
        return model, tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    def get_info(self) -> TrainingClientInfo:
        return TrainingClientInfo(
            session_id=self.session_id,
            base_model=self.model_config.base_model,
            lora_rank=self.lora_config.rank,
            step=self.step,
            metadata={
                "max_seq_length": self.model_config.max_seq_length,
                "dtype": self.model_config.dtype,
                **self.metadata,
            },
        )

    def register_custom_loss(self, name: str, loss_fn: CustomLoss) -> None:
        self._custom_losses[name] = loss_fn

    def _ensure_optimizer(self, params: AdamParams | None = None):
        import torch

        if self.optimizer is not None and params is None:
            return self.optimizer
        params = params or AdamParams(learning_rate=2e-5)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=params.learning_rate,
            betas=params.betas,
            eps=params.eps,
            weight_decay=params.weight_decay,
        )
        return self.optimizer

    def _batch_from_data(self, data: list[Datum]):
        import torch

        if not data:
            raise ValueError("data must not be empty")
        input_ids = [datum.model_input.to_ints() for datum in data]
        max_len = max(len(tokens) for tokens in input_ids)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
        padded = [tokens + [pad_token_id] * (max_len - len(tokens)) for tokens in input_ids]
        attention = [[1] * len(tokens) + [0] * (max_len - len(tokens)) for tokens in input_ids]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long, device=self.model.device),
            "attention_mask": torch.tensor(attention, dtype=torch.long, device=self.model.device),
        }

    def _labels_from_data(self, data: list[Datum], input_ids):
        import torch

        labels = []
        for datum, input_row in zip(data, input_ids):
            raw = datum.loss_fn_inputs.get("labels") or datum.loss_fn_inputs.get("target_tokens")
            if raw is None:
                labels.append(input_row.detach().clone())
            else:
                row = self._loss_tokens(raw)
                if len(row) < input_ids.shape[1]:
                    row = row + [-100] * (input_ids.shape[1] - len(row))
                labels.append(torch.tensor(row[: input_ids.shape[1]], dtype=torch.long, device=input_ids.device))
        return torch.stack(labels)

    def _loss_tokens(self, raw: Any) -> list[int]:
        if isinstance(raw, ModelInput):
            return raw.to_ints()
        if isinstance(raw, TensorData):
            raw = raw.data
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        return list(raw)

    def _cross_entropy_loss(self, data: list[Datum]):
        batch = self._batch_from_data(data)
        labels = self._labels_from_data(data, batch["input_ids"])
        outputs = self.model(**batch, labels=labels)
        return outputs.loss, outputs

    def _token_logprobs(self, tokens: list[int]) -> list[float | None]:
        import torch

        if len(tokens) < 2:
            return [None] * len(tokens)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            log_probs = outputs.logits[:, :-1, :].log_softmax(dim=-1)
            next_tokens = input_ids[:, 1:].unsqueeze(-1)
            gathered = torch.gather(log_probs, dim=-1, index=next_tokens).squeeze(0).squeeze(-1)
        return [None] + [float(value) for value in gathered.detach().cpu()]

    def compute_logprobs(self, prompt: ModelInput | list[int]) -> list[float | None]:
        tokens = prompt.to_ints() if isinstance(prompt, ModelInput) else list(prompt)
        return self._token_logprobs(tokens)

    def forward(self, data: list[Datum], loss_fn: str = "cross_entropy") -> ForwardOutput:
        import torch
        from unsloth import FastLanguageModel

        if loss_fn != "cross_entropy":
            raise UnsupportedLossError(f"Unsupported non-RL loss: {loss_fn}")
        FastLanguageModel.for_inference(self.model)
        with torch.no_grad():
            loss, _outputs = self._cross_entropy_loss(data)
        return ForwardOutput(loss=float(loss.detach().cpu()), metrics={"loss": float(loss.detach().cpu())})

    def forward_backward(
        self,
        data: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, Any] | None = None,
    ) -> ForwardBackwardOutput:
        del loss_fn_config
        import torch
        from unsloth import FastLanguageModel

        if loss_fn != "cross_entropy":
            raise UnsupportedLossError(
                f"Unsupported loss '{loss_fn}'. The MVP intentionally excludes RL losses."
            )
        FastLanguageModel.for_training(self.model)
        self._ensure_optimizer()
        loss, _outputs = self._cross_entropy_loss(data)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {float(loss.detach().cpu())}")
        loss.backward()
        value = float(loss.detach().cpu())
        return ForwardBackwardOutput(loss=value, metrics={"loss": value})

    def forward_backward_custom(
        self,
        data: list[Datum],
        loss_fn: str | CustomLoss,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> ForwardBackwardOutput:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_training(self.model)
        batch = self._batch_from_data(data)
        outputs = self.model(**batch)
        if isinstance(loss_fn, str):
            if loss_fn not in self._custom_losses:
                raise UnsupportedLossError(f"Custom loss is not registered: {loss_fn}")
            callable_loss = self._custom_losses[loss_fn]
        else:
            callable_loss = loss_fn
        loss, metrics = callable_loss(outputs, data, loss_fn_config or {})
        loss.backward()
        value = float(loss.detach().cpu())
        metrics = {"loss": value, **{key: float(item) for key, item in metrics.items()}}
        return ForwardBackwardOutput(loss=value, metrics=metrics)

    def optim_step(self, adam_params: AdamParams) -> OptimStepResult:
        import torch

        optimizer = self._ensure_optimizer(adam_params)
        trainable_parameters = [param for param in self.model.parameters() if param.requires_grad]
        if adam_params.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                adam_params.max_grad_norm,
                error_if_nonfinite=True,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return OptimStepResult(step=self.step, metrics={"step": float(self.step)})

    def sample(
        self,
        prompt: ModelInput | list[int],
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> SampleResponse:
        import torch
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(self.model)
        sampling_params = sampling_params or SamplingParams()
        prompt_tokens = prompt.to_ints() if isinstance(prompt, ModelInput) else list(prompt)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.model.device)
        stop_token_ids = self._stop_token_ids(sampling_params.stop)
        do_sample = sampling_params.temperature > 0
        generation_kwargs = {
            "max_new_tokens": sampling_params.max_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_samples,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "remove_invalid_values": True,
            "renormalize_logits": True,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if stop_token_ids:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class StopOnTokenSequences(StoppingCriteria):
                def __init__(self, stop_sequences: list[list[int]], prompt_length: int):
                    self.stop_sequences = stop_sequences
                    self.prompt_length = prompt_length

                def __call__(self, input_ids, scores, **kwargs) -> bool:
                    del scores, kwargs
                    if input_ids.shape[1] <= self.prompt_length:
                        return False
                    for row in input_ids:
                        generated = row[self.prompt_length :].tolist()
                        if not any(
                            len(generated) >= len(stop) and generated[-len(stop) :] == stop
                            for stop in self.stop_sequences
                        ):
                            return False
                    return True

            generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [StopOnTokenSequences(stop_token_ids, len(prompt_tokens))]
            )
        if do_sample:
            generation_kwargs["temperature"] = sampling_params.temperature
            generation_kwargs["top_p"] = sampling_params.top_p
        if do_sample and sampling_params.top_k is not None and sampling_params.top_k > 0:
            generation_kwargs["top_k"] = sampling_params.top_k
        if sampling_params.seed is not None:
            torch.manual_seed(sampling_params.seed)
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, **generation_kwargs)
        generated_logprobs = self._generated_logprobs(outputs)
        sequences = []
        for index, output in enumerate(outputs.sequences):
            tokens = output.detach().cpu().tolist()
            completion_tokens = tokens[len(prompt_tokens) :]
            completion_tokens, finish_reason = self._trim_stop_tokens(completion_tokens, stop_token_ids)
            if finish_reason != "stop":
                finish_reason = self._finish_reason(completion_tokens, sampling_params.max_tokens)
            text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
            sequence_logprobs = generated_logprobs[index][: len(completion_tokens)] if generated_logprobs else None
            sequences.append(
                GeneratedSequence(
                    tokens=completion_tokens,
                    text=text,
                    logprobs=sequence_logprobs,
                    finish_reason=finish_reason,
                    stop_reason=finish_reason,
                )
            )
        prompt_logprobs, topk = self._prompt_logprobs(
            prompt_tokens,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
        )
        return SampleResponse(
            sequences=sequences,
            prompt_logprobs=prompt_logprobs,
            topk_prompt_logprobs=topk,
        )

    def _stop_token_ids(self, stops: list[str]) -> list[list[int]]:
        tokenized = []
        for stop in stops:
            tokens = self.tokenizer.encode(stop, add_special_tokens=False)
            if tokens:
                tokenized.append(tokens)
        return tokenized

    def _trim_stop_tokens(self, tokens: list[int], stop_token_ids: list[list[int]]) -> tuple[list[int], str | None]:
        for stop in stop_token_ids:
            if len(tokens) >= len(stop) and tokens[-len(stop) :] == stop:
                return tokens[: -len(stop)], "stop"
        return tokens, None

    def _finish_reason(self, tokens: list[int], max_tokens: int) -> str:
        eos_token_id = self.tokenizer.eos_token_id
        if tokens and eos_token_id is not None and tokens[-1] == eos_token_id:
            return "eos"
        if len(tokens) >= max_tokens:
            return "length"
        return "stop"

    def _generated_logprobs(self, outputs) -> list[list[float | None]] | None:
        import torch

        scores = getattr(outputs, "scores", None)
        sequences = getattr(outputs, "sequences", None)
        if not scores or sequences is None:
            return None
        prompt_length = sequences.shape[1] - len(scores)
        all_logprobs: list[list[float | None]] = [[] for _ in range(sequences.shape[0])]
        for step, score in enumerate(scores):
            log_probs = score.log_softmax(dim=-1)
            token_ids = sequences[:, prompt_length + step]
            gathered = torch.gather(log_probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
            for row_index, value in enumerate(gathered.detach().cpu()):
                all_logprobs[row_index].append(float(value))
        return all_logprobs

    def _prompt_logprobs(
        self,
        prompt_tokens: list[int],
        *,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int,
    ) -> tuple[list[float | None] | None, list[list[tuple[int, float]]] | None]:
        import torch

        if not include_prompt_logprobs and topk_prompt_logprobs <= 0:
            return None, None
        if len(prompt_tokens) < 2:
            empty_logprobs = [None] * len(prompt_tokens) if include_prompt_logprobs else None
            empty_topk = [[] for _ in prompt_tokens] if topk_prompt_logprobs > 0 else None
            return empty_logprobs, empty_topk
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            log_probs = outputs.logits[:, :-1, :].log_softmax(dim=-1).squeeze(0)
        prompt_logprobs = None
        if include_prompt_logprobs:
            next_tokens = input_ids[:, 1:].squeeze(0)
            gathered = torch.gather(log_probs, dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)
            prompt_logprobs = [None] + [float(value) for value in gathered.detach().cpu()]
        topk = None
        if topk_prompt_logprobs > 0:
            values, indices = torch.topk(log_probs, k=topk_prompt_logprobs, dim=-1)
            topk = [[]]
            for row_values, row_indices in zip(values.detach().cpu(), indices.detach().cpu()):
                topk.append([(int(token_id), float(value)) for token_id, value in zip(row_indices, row_values)])
        return prompt_logprobs, topk

    def _save_weights(self, target: Path, *, include_optimizer: bool, kind: str) -> CheckpointRef:
        import torch

        with atomic_checkpoint_dir(target) as tmp_dir:
            self.model.save_pretrained(str(tmp_dir))
            self.tokenizer.save_pretrained(str(tmp_dir))
            if include_optimizer:
                optimizer = self._ensure_optimizer()
                torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
            manifest = base_manifest(
                kind=kind,
                step=self.step,
                base_model=self.model_config.base_model,
                lora=asdict(self.lora_config),
                has_optimizer=include_optimizer,
                extra={
                    "session_id": self.session_id,
                    "model_path": str(target),
                    "metadata": self.metadata,
                },
            )
            write_manifest(tmp_dir, manifest)
        return checkpoint_ref(target, has_optimizer=include_optimizer)

    def save_state(self, path: str | None = None, include_optimizer: bool = False) -> CheckpointRef:
        target = resolve_path(path) if path else new_checkpoint_path(self.checkpoint_root, "state", self.step)
        return self._save_weights(target, include_optimizer=include_optimizer, kind="training_state")

    def load_state(self, path: str, with_optimizer: bool = False) -> None:
        import torch

        manifest = read_manifest(path)
        self.step = int(manifest.get("step", 0))
        resolved = resolve_path(path)
        adapter_name = f"ray_unsloth_step_{self.step}"
        self.model.load_adapter(str(resolved), adapter_name=adapter_name, is_trainable=True)
        if hasattr(self.model, "set_adapter"):
            self.model.set_adapter(adapter_name)
        optimizer_path = resolved / "optimizer.pt"
        if with_optimizer:
            if not optimizer_path.exists():
                raise FileNotFoundError(f"Checkpoint does not include optimizer state: {optimizer_path}")
            optimizer = self._ensure_optimizer()
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.model.device))

    def load_state_with_optimizer(self, path: str) -> None:
        self.load_state(path, with_optimizer=True)

    def save_weights_for_sampler(self, path: str | None = None) -> SaveWeightsForSamplerResponse:
        target = resolve_path(path) if path else new_checkpoint_path(self.checkpoint_root, "sampler", self.step)
        checkpoint = self._save_weights(target, include_optimizer=False, kind="sampler_weights")
        return SaveWeightsForSamplerResponse(path=checkpoint.path, checkpoint=checkpoint)
