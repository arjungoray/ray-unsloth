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
    ) -> None:
        self.session_id = session_id
        self.model_config = model_config
        self.lora_config = lora_config
        self.checkpoint_root = checkpoint_root
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
                row = list(raw)
                if len(row) < input_ids.shape[1]:
                    row = row + [-100] * (input_ids.shape[1] - len(row))
                labels.append(torch.tensor(row[: input_ids.shape[1]], dtype=torch.long, device=input_ids.device))
        return torch.stack(labels)

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
        from unsloth import FastLanguageModel

        if loss_fn != "cross_entropy":
            raise UnsupportedLossError(
                f"Unsupported loss '{loss_fn}'. The MVP intentionally excludes RL losses."
            )
        FastLanguageModel.for_training(self.model)
        self._ensure_optimizer()
        loss, _outputs = self._cross_entropy_loss(data)
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
        if adam_params.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), adam_params.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return OptimStepResult(step=self.step, metrics={"step": float(self.step)})

    def sample(
        self,
        prompt: ModelInput | list[int],
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
    ) -> SampleResponse:
        import torch
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(self.model)
        sampling_params = sampling_params or SamplingParams()
        prompt_tokens = prompt.to_ints() if isinstance(prompt, ModelInput) else list(prompt)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.model.device)
        do_sample = sampling_params.temperature > 0
        generation_kwargs = {
            "max_new_tokens": sampling_params.max_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_samples,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "remove_invalid_values": True,
            "renormalize_logits": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = sampling_params.temperature
            generation_kwargs["top_p"] = sampling_params.top_p
        if do_sample and sampling_params.top_k is not None:
            generation_kwargs["top_k"] = sampling_params.top_k
        if sampling_params.seed is not None:
            torch.manual_seed(sampling_params.seed)
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, **generation_kwargs)
        sequences = []
        for output in outputs:
            tokens = output.detach().cpu().tolist()
            text = self.tokenizer.decode(tokens[len(prompt_tokens) :], skip_special_tokens=True)
            sequences.append(GeneratedSequence(tokens=tokens, text=text))
        return SampleResponse(sequences=sequences)

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
