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
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        backend: str = "nccl",
        init_method: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.model_config = model_config
        self.lora_config = lora_config
        self.checkpoint_root = checkpoint_root
        self.metadata = dict(metadata or {})
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.backend = backend
        self.init_method = init_method
        self.step = 0
        self.optimizer = None
        self._custom_losses: dict[str, CustomLoss] = {}

        self._setup_distributed()
        self.model, self.tokenizer = self._load_model(model_path=model_path)
        self._wrap_distributed_model()
        if model_path:
            self.load_state(model_path, with_optimizer=with_optimizer)

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0

    def _setup_distributed(self) -> None:
        if not self.is_distributed:
            return
        if self.init_method is None:
            raise ValueError("init_method is required when world_size > 1.")
        import torch
        import torch.distributed as dist

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                init_method=self.init_method,
                rank=self.rank,
                world_size=self.world_size,
            )

    def _wrap_distributed_model(self) -> None:
        if not self.is_distributed:
            return
        import torch
        from torch.nn.parallel import DistributedDataParallel

        device_ids = [self.local_rank] if torch.cuda.is_available() else None
        output_device = self.local_rank if torch.cuda.is_available() else None
        self.model = DistributedDataParallel(
            self.model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=True,
            broadcast_bucket_size=25 * 1024 * 1024,
        )

    def _unwrap_model(self):
        return getattr(self.model, "module", self.model)

    def _model_device(self):
        import torch

        model = self._unwrap_model()
        device = getattr(model, "device", None)
        if device is not None:
            return device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    def _barrier(self) -> None:
        if not self.is_distributed:
            return
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()

    def _load_model(self, model_path: str | None = None):
        from unsloth import FastLanguageModel

        model_name = self.model_config.base_model
        loader_kwargs: dict[str, Any] = {}
        if self.model_config.device_map is not None:
            loader_kwargs["device_map"] = self.model_config.device_map
        elif self.is_distributed:
            # Each Ray DDP worker gets exactly one visible GPU, so local ordinal
            # zero is the only valid CUDA device inside the worker process.
            loader_kwargs["device_map"] = {"": self.local_rank}
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.model_config.max_seq_length,
            dtype=_torch_dtype(self.model_config.dtype),
            load_in_4bit=self.model_config.load_in_4bit,
            fast_inference=self.model_config.fast_inference,
            gpu_memory_utilization=self.model_config.gpu_memory_utilization,
            trust_remote_code=self.model_config.trust_remote_code,
            **loader_kwargs,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_config.rank,
            target_modules=self.lora_config.target_modules,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            bias=self.lora_config.bias,
            use_gradient_checkpointing=self.lora_config.use_gradient_checkpointing,
            random_state=self.lora_config.random_state,
            use_rslora=self.lora_config.use_rslora,
            loftq_config=self.lora_config.loftq_config,
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

        if self.optimizer is not None:
            if params is not None:
                for group in self.optimizer.param_groups:
                    group["lr"] = params.learning_rate
                    group["betas"] = params.betas
                    group["eps"] = params.eps
                    group["weight_decay"] = params.weight_decay
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
            "input_ids": torch.tensor(padded, dtype=torch.long, device=self._model_device()),
            "attention_mask": torch.tensor(attention, dtype=torch.long, device=self._model_device()),
        }

    def _loss_tensor_data(self, raw: Any) -> list[Any]:
        if isinstance(raw, ModelInput):
            return raw.to_ints()
        if isinstance(raw, TensorData):
            raw = raw.tolist()
        elif hasattr(raw, "tolist"):
            raw = raw.tolist()
        return list(raw)

    def _label_rows_from_data(self, data: list[Datum], input_ids):
        import torch

        rows = []
        for datum, input_row in zip(data, input_ids):
            raw = datum.loss_fn_inputs.get("labels") or datum.loss_fn_inputs.get("target_tokens")
            if raw is None:
                row = input_row.detach().clone().tolist()
            else:
                row = [int(value) for value in self._loss_tensor_data(raw)]
            if len(row) < input_ids.shape[1]:
                row = row + [-100] * (input_ids.shape[1] - len(row))
            rows.append(torch.tensor(row[: input_ids.shape[1]], dtype=torch.long, device=input_ids.device))
        return torch.stack(rows)

    def _loss_tokens(self, raw: Any) -> list[int]:
        if isinstance(raw, ModelInput):
            return raw.to_ints()
        if isinstance(raw, TensorData):
            raw = raw.tolist()
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        return list(raw)

    def _cross_entropy_loss(self, data: list[Datum]):
        import torch

        batch = self._batch_from_data(data)
        outputs = self.model(**batch)
        log_probs = outputs.logits.log_softmax(dim=-1)

        external_logprobs = torch.zeros(
            batch["input_ids"].shape,
            dtype=log_probs.dtype,
            device=log_probs.device,
        )
        total_loss = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
        total_weight = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)

        for row_idx, datum in enumerate(data):
            input_len = int(batch["attention_mask"][row_idx].sum().detach().cpu())
            has_target_tokens = "target_tokens" in datum.loss_fn_inputs
            raw_targets = datum.loss_fn_inputs.get("target_tokens")
            raw_labels = datum.loss_fn_inputs.get("labels")
            raw_weights = datum.loss_fn_inputs.get("weights")

            if has_target_tokens and raw_targets is not None:
                targets = [int(value) for value in self._loss_tensor_data(raw_targets)][:input_len]
                positions = list(range(min(input_len, len(targets))))
                output_indices = positions
            else:
                labels = (
                    [int(value) for value in self._loss_tensor_data(raw_labels)][:input_len]
                    if raw_labels is not None
                    else batch["input_ids"][row_idx, :input_len].detach().cpu().tolist()
                )
                targets = labels
                positions = list(range(1, min(input_len, len(labels))))
                output_indices = positions

            weights = None
            if raw_weights is not None:
                weights = [float(value) for value in self._loss_tensor_data(raw_weights)][:input_len]

            for output_index, target_index in zip(output_indices, positions):
                target_token = int(targets[target_index])
                if target_token == -100:
                    continue
                weight = 1.0
                if weights is not None:
                    weight = float(weights[target_index])
                if weight == 0.0:
                    continue
                logit_index = target_index if has_target_tokens else target_index - 1
                token_logprob = log_probs[row_idx, logit_index, target_token]
                external_logprobs[row_idx, output_index] = token_logprob
                total_loss = total_loss - token_logprob * weight
                total_weight = total_weight + weight

        loss = total_loss
        return loss, outputs, external_logprobs

    def _policy_loss(
        self,
        data: list[Datum],
        *,
        loss_fn: str,
        loss_fn_config: dict[str, Any] | None = None,
    ):
        import torch

        loss_fn_config = loss_fn_config or {}
        batch = self._batch_from_data(data)
        outputs = self.model(**batch)
        log_probs = outputs.logits.log_softmax(dim=-1)
        external_logprobs = torch.zeros(
            batch["input_ids"].shape,
            dtype=log_probs.dtype,
            device=log_probs.device,
        )
        external_ratios = torch.zeros_like(external_logprobs)
        total_loss = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
        clip_low = float(loss_fn_config.get("clip_low_threshold", 0.8))
        clip_high = float(loss_fn_config.get("clip_high_threshold", 1.2))

        for row_idx, datum in enumerate(data):
            input_len = int(batch["attention_mask"][row_idx].sum().detach().cpu())
            targets = [int(value) for value in self._loss_tensor_data(datum.loss_fn_inputs["target_tokens"])][:input_len]
            old_logprobs = [float(value) for value in self._loss_tensor_data(datum.loss_fn_inputs["logprobs"])][:input_len]
            advantages = [float(value) for value in self._loss_tensor_data(datum.loss_fn_inputs["advantages"])][:input_len]
            raw_weights = datum.loss_fn_inputs.get("weights")
            weights = [1.0] * input_len
            if raw_weights is not None:
                weights = [float(value) for value in self._loss_tensor_data(raw_weights)][:input_len]

            for pos, target_token in enumerate(targets):
                if target_token == -100 or pos >= input_len:
                    continue
                weight = weights[pos] if pos < len(weights) else 1.0
                advantage = advantages[pos] if pos < len(advantages) else 0.0
                if weight == 0.0 or advantage == 0.0:
                    continue
                current_logprob = log_probs[row_idx, pos, target_token]
                old_logprob = old_logprobs[pos] if pos < len(old_logprobs) else 0.0
                ratio = torch.exp(current_logprob - old_logprob)
                external_logprobs[row_idx, pos] = current_logprob
                external_ratios[row_idx, pos] = ratio
                weighted_advantage = weight * advantage
                if loss_fn == "importance_sampling":
                    token_loss = -ratio * weighted_advantage
                elif loss_fn == "ppo":
                    clipped_ratio = ratio.clamp(clip_low, clip_high)
                    token_loss = -torch.minimum(ratio * weighted_advantage, clipped_ratio * weighted_advantage)
                elif loss_fn == "cispo":
                    clipped_ratio = ratio.detach().clamp(clip_low, clip_high)
                    token_loss = -clipped_ratio * weighted_advantage * current_logprob
                else:
                    raise UnsupportedLossError(f"Unsupported loss: {loss_fn}")
                total_loss = total_loss + token_loss

        return total_loss, outputs, external_logprobs, external_ratios

    def _loss_fn_outputs(
        self,
        row_logprobs: Any,
        *,
        ratios: Any | None = None,
    ) -> list[dict[str, TensorData]]:
        rows: list[dict[str, TensorData]] = []
        for row_idx in range(row_logprobs.shape[0]):
            values = [float(value) for value in row_logprobs[row_idx].detach().cpu()]
            output = {
                "logprobs": TensorData(
                    data=values,
                    dtype=str(row_logprobs.dtype).removeprefix("torch."),
                    shape=[len(values)],
                )
            }
            if ratios is not None:
                ratio_values = [float(value) for value in ratios[row_idx].detach().cpu()]
                output["ratios"] = TensorData(
                    data=ratio_values,
                    dtype=str(ratios.dtype).removeprefix("torch."),
                    shape=[len(ratio_values)],
                )
            rows.append(output)
        return rows

    def _token_logprobs(self, tokens: list[int]) -> list[float | None]:
        import torch

        if len(tokens) < 2:
            return [None] * len(tokens)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self._model_device())
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

        FastLanguageModel.for_inference(self._unwrap_model())
        with torch.no_grad():
            if loss_fn == "cross_entropy":
                loss, _outputs, row_logprobs = self._cross_entropy_loss(data)
                loss_fn_outputs = self._loss_fn_outputs(row_logprobs)
            elif loss_fn in {"importance_sampling", "ppo", "cispo"}:
                loss, _outputs, row_logprobs, ratios = self._policy_loss(data, loss_fn=loss_fn)
                loss_fn_outputs = self._loss_fn_outputs(row_logprobs, ratios=ratios)
            else:
                raise UnsupportedLossError(f"Unsupported loss: {loss_fn}")
        value = float(loss.detach().cpu())
        return ForwardOutput(loss=value, metrics={"loss": value, "loss:sum": value}, loss_fn_outputs=loss_fn_outputs)

    def forward_backward(
        self,
        data: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, Any] | None = None,
    ) -> ForwardBackwardOutput:
        import torch
        from unsloth import FastLanguageModel

        FastLanguageModel.for_training(self._unwrap_model())
        self._ensure_optimizer()
        if loss_fn == "cross_entropy":
            loss, _outputs, row_logprobs = self._cross_entropy_loss(data)
            loss_fn_outputs = self._loss_fn_outputs(row_logprobs)
        elif loss_fn in {"importance_sampling", "ppo", "cispo"}:
            loss, _outputs, row_logprobs, ratios = self._policy_loss(
                data,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            )
            loss_fn_outputs = self._loss_fn_outputs(row_logprobs, ratios=ratios)
        else:
            raise UnsupportedLossError(f"Unsupported loss '{loss_fn}'.")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {float(loss.detach().cpu())}")
        backward_loss = loss * self.world_size if self.is_distributed else loss
        backward_loss.backward()
        value = float(loss.detach().cpu())
        return ForwardBackwardOutput(loss=value, metrics={"loss": value, "loss:sum": value}, loss_fn_outputs=loss_fn_outputs)

    def forward_backward_custom(
        self,
        data: list[Datum],
        loss_fn: str | CustomLoss,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> ForwardBackwardOutput:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_training(self._unwrap_model())
        batch = self._batch_from_data(data)
        outputs = self.model(**batch)
        if isinstance(loss_fn, str):
            if loss_fn not in self._custom_losses:
                raise UnsupportedLossError(f"Custom loss is not registered: {loss_fn}")
            callable_loss = self._custom_losses[loss_fn]
        else:
            callable_loss = loss_fn
        loss, metrics = callable_loss(outputs, data, loss_fn_config or {})
        backward_loss = loss * self.world_size if self.is_distributed else loss
        backward_loss.backward()
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

        FastLanguageModel.for_inference(self._unwrap_model())
        sampling_params = sampling_params or SamplingParams()
        prompt_tokens = prompt.to_ints() if isinstance(prompt, ModelInput) else list(prompt)
        stop_token_ids = self._stop_token_ids(sampling_params.stop)
        if sampling_params.seed is not None:
            torch.manual_seed(sampling_params.seed)

        generated_outputs = self._generate(
            prompt_tokens=prompt_tokens,
            num_samples=num_samples,
            sampling_params=sampling_params,
            stop_token_ids=stop_token_ids,
        )
        sequences = []
        for completion_tokens, sequence_logprobs, finish_reason in generated_outputs:
            text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
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

    def _generate(
        self,
        *,
        prompt_tokens: list[int],
        num_samples: int,
        sampling_params: SamplingParams,
        stop_token_ids: list[list[int]],
    ) -> list[tuple[list[int], list[float | None], str]]:
        if hasattr(self._unwrap_model(), "generate"):
            try:
                return self._generate_with_transformers_generate(
                    prompt_tokens=prompt_tokens,
                    num_samples=num_samples,
                    sampling_params=sampling_params,
                    stop_token_ids=stop_token_ids,
                )
            except (AttributeError, TypeError, ValueError):
                pass
        return self._generate_with_forward_loop(
            prompt_tokens=prompt_tokens,
            num_samples=num_samples,
            sampling_params=sampling_params,
            stop_token_ids=stop_token_ids,
        )

    def _generate_with_transformers_generate(
        self,
        *,
        prompt_tokens: list[int],
        num_samples: int,
        sampling_params: SamplingParams,
        stop_token_ids: list[list[int]],
    ) -> list[tuple[list[int], list[float | None], str]]:
        import torch

        model = self._unwrap_model()
        device = self._model_device()
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": sampling_params.max_tokens,
            "num_return_sequences": num_samples,
            "do_sample": sampling_params.temperature > 0,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": pad_token_id,
        }
        if sampling_params.temperature > 0:
            kwargs["temperature"] = sampling_params.temperature
            kwargs["top_p"] = sampling_params.top_p
            if sampling_params.top_k is not None:
                kwargs["top_k"] = sampling_params.top_k
        if self.tokenizer.eos_token_id is not None:
            kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if sampling_params.seed is not None:
            torch.manual_seed(sampling_params.seed)

        with torch.no_grad():
            generated = model.generate(**kwargs)

        sequences = generated.sequences.detach().cpu().tolist()
        outputs = []
        prompt_length = len(prompt_tokens)
        for row_index, sequence in enumerate(sequences):
            completion_tokens = list(sequence[prompt_length:])
            finish_reason = self._finish_reason(completion_tokens, sampling_params.max_tokens)
            trimmed_tokens, stop_reason = self._trim_at_first_stop(completion_tokens, stop_token_ids)
            if stop_reason is not None:
                finish_reason = stop_reason
            outputs.append((trimmed_tokens, [], finish_reason))
        logprob_rows = self._completion_logprobs_batch(prompt_tokens, [tokens for tokens, _logprobs, _reason in outputs])
        return [
            (tokens, logprobs, finish_reason)
            for (tokens, _empty_logprobs, finish_reason), logprobs in zip(outputs, logprob_rows)
        ]

    def _completion_logprobs_batch(
        self,
        prompt_tokens: list[int],
        completion_rows: list[list[int]],
    ) -> list[list[float | None]]:
        import torch

        if not completion_rows:
            return []
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
        full_rows = [prompt_tokens + completion for completion in completion_rows]
        max_len = max(len(tokens) for tokens in full_rows)
        padded = [tokens + [pad_token_id] * (max_len - len(tokens)) for tokens in full_rows]
        attention = [[1] * len(tokens) + [0] * (max_len - len(tokens)) for tokens in full_rows]
        input_ids = torch.tensor(padded, dtype=torch.long, device=self._model_device())
        attention_mask = torch.tensor(attention, dtype=torch.long, device=self._model_device())
        with torch.no_grad():
            model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = model_outputs.logits.log_softmax(dim=-1)

        prompt_len = len(prompt_tokens)
        rows: list[list[float | None]] = []
        for row_index, completion in enumerate(completion_rows):
            values: list[float | None] = []
            for token_index, token in enumerate(completion):
                logit_index = prompt_len + token_index - 1
                if logit_index < 0:
                    values.append(None)
                else:
                    values.append(float(log_probs[row_index, logit_index, int(token)].detach().cpu()))
            rows.append(values)
        return rows

    def _generate_with_forward_loop(
        self,
        *,
        prompt_tokens: list[int],
        num_samples: int,
        sampling_params: SamplingParams,
        stop_token_ids: list[list[int]],
    ) -> list[tuple[list[int], list[float | None], str]]:
        import torch

        outputs = []
        for _sample_index in range(num_samples):
            context = list(prompt_tokens)
            completion_tokens: list[int] = []
            logprobs: list[float | None] = []
            finish_reason: str | None = None

            for _step in range(sampling_params.max_tokens):
                input_ids = torch.tensor([context], dtype=torch.long, device=self._model_device())
                attention_mask = torch.tensor([[1] * len(context)], dtype=torch.long, device=self._model_device())
                with torch.no_grad():
                    model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = model_outputs.logits[:, -1, :].squeeze(0).float()
                logits = torch.nan_to_num(logits, nan=-1e30, posinf=1e30, neginf=-1e30)
                next_token, logprob = self._next_sampled_token(logits, sampling_params)

                completion_tokens.append(next_token)
                logprobs.append(logprob)
                context.append(next_token)

                if self._is_eos_token(next_token):
                    finish_reason = "eos"
                    break
                if self._ends_with_stop(completion_tokens, stop_token_ids):
                    finish_reason = "stop"
                    break

            if finish_reason == "stop":
                trimmed_tokens, _stop_reason = self._trim_stop_tokens(completion_tokens, stop_token_ids)
                logprobs = logprobs[: len(trimmed_tokens)]
                completion_tokens = trimmed_tokens
            if finish_reason is None:
                finish_reason = "length" if len(completion_tokens) >= sampling_params.max_tokens else "stop"
            outputs.append((completion_tokens, logprobs, finish_reason))
        return outputs

    def _next_sampled_token(self, logits, sampling_params: SamplingParams) -> tuple[int, float]:
        import torch

        log_probs = logits.log_softmax(dim=-1)
        if sampling_params.temperature <= 0:
            token = int(torch.argmax(logits).detach().cpu())
            return token, float(log_probs[token].detach().cpu())

        filtered_logits = logits / sampling_params.temperature
        filtered_logits = self._apply_top_k_top_p(
            filtered_logits,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
        )
        probs = torch.softmax(filtered_logits, dim=-1)
        token = int(torch.multinomial(probs, num_samples=1).detach().cpu())
        return token, float(log_probs[token].detach().cpu())

    def _apply_top_k_top_p(self, logits, *, top_k: int | None, top_p: float):
        import torch

        filtered = logits.clone()
        if top_k is not None and top_k > 0 and top_k < filtered.numel():
            values, _indices = torch.topk(filtered, k=top_k)
            filtered = filtered.masked_fill(filtered < values[-1], -float("inf"))

        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            remove = sorted_probs.cumsum(dim=-1) > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            filtered[sorted_indices[remove]] = -float("inf")

        if torch.isinf(filtered).all():
            return logits
        return filtered

    def _is_eos_token(self, token: int) -> bool:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        return eos_token_id is not None and token == eos_token_id

    def _ends_with_stop(self, tokens: list[int], stop_token_ids: list[list[int]]) -> bool:
        return any(len(tokens) >= len(stop) and tokens[-len(stop) :] == stop for stop in stop_token_ids)

    def _encode_text(self, text: str, *, add_special_tokens: bool) -> list[int]:
        text_tokenizer = getattr(self.tokenizer, "tokenizer", None)
        if text_tokenizer is not None and hasattr(text_tokenizer, "encode"):
            return list(text_tokenizer.encode(text, add_special_tokens=add_special_tokens))
        if hasattr(self.tokenizer, "encode"):
            return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

        try:
            encoded = self.tokenizer(text=text, add_special_tokens=add_special_tokens)
        except TypeError:
            encoded = self.tokenizer(text, add_special_tokens=add_special_tokens)
        if isinstance(encoded, dict):
            tokens = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            tokens = encoded.input_ids
        else:
            tokens = encoded
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if tokens and isinstance(tokens[0], list):
            if len(tokens) != 1:
                raise ValueError("Expected a single tokenized stop sequence")
            tokens = tokens[0]
        return list(tokens)

    def _stop_token_ids(self, stops: list[str]) -> list[list[int]]:
        tokenized = []
        for stop in stops:
            tokens = self._encode_text(stop, add_special_tokens=False)
            if tokens:
                tokenized.append(tokens)
        return tokenized

    def _trim_stop_tokens(self, tokens: list[int], stop_token_ids: list[list[int]]) -> tuple[list[int], str | None]:
        for stop in stop_token_ids:
            if len(tokens) >= len(stop) and tokens[-len(stop) :] == stop:
                return tokens[: -len(stop)], "stop"
        return tokens, None

    def _trim_at_first_stop(self, tokens: list[int], stop_token_ids: list[list[int]]) -> tuple[list[int], str | None]:
        for end_index in range(1, len(tokens) + 1):
            prefix = tokens[:end_index]
            trimmed, reason = self._trim_stop_tokens(prefix, stop_token_ids)
            if reason is not None:
                return trimmed, reason
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
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self._model_device())
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
            self._unwrap_model().save_pretrained(str(tmp_dir))
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
        checkpoint = None
        try:
            if self.is_rank0:
                checkpoint = self._save_weights(target, include_optimizer=include_optimizer, kind="training_state")
        finally:
            self._barrier()
        return checkpoint

    def load_state(self, path: str, with_optimizer: bool = False) -> None:
        import torch
        import torch.distributed as dist

        manifest = read_manifest(path) if self.is_rank0 else None
        if self.is_distributed:
            values = [manifest]
            dist.broadcast_object_list(values, src=0)
            manifest = values[0]
        if manifest is None:
            raise RuntimeError("Rank 0 did not provide checkpoint manifest.")
        self.step = int(manifest.get("step", 0))
        resolved = resolve_path(path)
        adapter_name = f"ray_unsloth_step_{self.step}"
        model = self._unwrap_model()
        model.load_adapter(str(resolved), adapter_name=adapter_name, is_trainable=True)
        if hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
        optimizer_path = resolved / "optimizer.pt"
        if with_optimizer:
            if not optimizer_path.exists():
                raise FileNotFoundError(f"Checkpoint does not include optimizer state: {optimizer_path}")
            optimizer = self._ensure_optimizer()
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=self._model_device()))
        self._barrier()

    def load_state_with_optimizer(self, path: str) -> None:
        self.load_state(path, with_optimizer=True)

    def save_weights_for_sampler(self, path: str | None = None) -> SaveWeightsForSamplerResponse:
        target = resolve_path(path) if path else new_checkpoint_path(self.checkpoint_root, "sampler", self.step)
        checkpoint = None
        try:
            if self.is_rank0:
                checkpoint = self._save_weights(target, include_optimizer=False, kind="sampler_weights")
        finally:
            self._barrier()
        if checkpoint is None:
            return None
        return SaveWeightsForSamplerResponse(path=checkpoint.path, checkpoint=checkpoint)
