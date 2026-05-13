"""GPU-local Unsloth model operations."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ray_unsloth.checkpoints import (
    atomic_checkpoint_dir,
    base_manifest,
    checkpoint_ref,
    new_checkpoint_path,
    read_manifest,
    resolve_path,
    write_manifest,
)
from ray_unsloth.config import LoRAConfig, ModelConfig, SpeedConfig
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


def _requires_transformers_5_for_qwen3_5(base_model: str) -> bool:
    normalized = base_model.lower().replace("-", "_").replace("/", "_").replace(".", "_")
    return "qwen3_5" in normalized


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _flash_attention_2_available() -> bool:
    return _module_available("flash_attn")


def _flash_attention_3_available() -> bool:
    return _module_available("flash_attn_interface")


@dataclass(slots=True)
class _BatchPlan:
    batch: dict[str, Any]
    row_lengths: list[int]
    row_starts: list[int]
    packed: bool = False

    @property
    def max_len(self) -> int:
        return max(self.row_lengths) if self.row_lengths else 0


class UnslothEngine:
    """Owns one Unsloth model, tokenizer, optimizer, and gradient state."""

    def __init__(
        self,
        *,
        session_id: str,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        checkpoint_root: str,
        speed_config: SpeedConfig | None = None,
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
        self.speed_config = speed_config or SpeedConfig()
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
        self._configure_unsloth_environment()
        from unsloth import FastLanguageModel

        model_name = self.model_config.base_model
        loader_kwargs: dict[str, Any] = {}
        if self.model_config.device_map is not None:
            loader_kwargs["device_map"] = self.model_config.device_map
        elif self.is_distributed:
            # Each Ray DDP worker gets exactly one visible GPU, so local ordinal
            # zero is the only valid CUDA device inside the worker process.
            loader_kwargs["device_map"] = {"": self.local_rank}
        attn_implementation = self._effective_attn_implementation()
        self._resolved_attn_implementation = attn_implementation
        if attn_implementation is not None:
            loader_kwargs["attn_implementation"] = attn_implementation
        fast_inference = self._effective_fast_inference()
        if fast_inference:
            loader_kwargs["max_lora_rank"] = self.lora_config.rank
            if self._effective_vllm_standby():
                loader_kwargs["unsloth_vllm_standby"] = True
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.model_config.max_seq_length,
            dtype=_torch_dtype(self.model_config.dtype),
            load_in_4bit=self.model_config.load_in_4bit,
            fast_inference=fast_inference,
            gpu_memory_utilization=self._effective_gpu_memory_utilization(fast_inference),
            trust_remote_code=self.model_config.trust_remote_code,
            **loader_kwargs,
        )
        self._set_attention_implementation(model)
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
        self._set_attention_implementation(model)
        return model, tokenizer

    def _speed(self) -> SpeedConfig:
        return getattr(self, "speed_config", SpeedConfig())

    def _effective_fast_inference(self) -> bool:
        value = self.model_config.fast_inference
        if value == "auto":
            if _requires_transformers_5_for_qwen3_5(self.model_config.base_model):
                return False
            return not bool(self.model_config.trust_remote_code)
        return bool(value)

    def _effective_vllm_standby(self) -> bool:
        value = self._speed().vllm_standby
        if value == "auto":
            return self._effective_fast_inference()
        return bool(value)

    def _effective_gpu_memory_utilization(self, fast_inference: bool) -> float:
        if fast_inference and self._effective_vllm_standby() and self.model_config.gpu_memory_utilization == 0.85:
            return 0.95
        return self.model_config.gpu_memory_utilization

    def _effective_attn_implementation(self) -> str | None:
        configured = self.model_config.attn_implementation
        if configured not in {None, "auto"}:
            return configured
        value = self._speed().flash_attention_2
        if value is False:
            return None
        if value is True:
            return "flash_attention_2"
        # auto: pick best available backend based on GPU capability
        return self._best_attn_backend()

    def _gpu_compute_capability(self) -> tuple[int, int] | None:
        """Return (major, minor) compute capability or None if unavailable."""
        try:
            import torch
        except Exception:
            return None
        try:
            if not torch.cuda.is_available():
                return None
            return torch.cuda.get_device_capability()
        except Exception:
            return None

    def _best_attn_backend(self) -> str | None:
        """Select the best attention backend for the current GPU.

        Tiered selection based on GPU architecture and installed kernels:
          - compute capability >= 9.0 (H100/B200):  flash_attention_3 when available
          - compute capability == 8.0 (A100):        flash_attention_2 when available
          - all other GPUs or missing FA kernels:    xformers
        """
        cap = self._gpu_compute_capability()
        if cap is None:
            return None
        major, minor = cap

        if major >= 9 and _flash_attention_3_available():
            return "flash_attention_3"

        if major == 8 and minor == 0 and _flash_attention_2_available():
            return "flash_attention_2"

        return "xformers"

    def _configure_unsloth_environment(self) -> None:
        if self._effective_vllm_standby():
            os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

    def _padding_free_requested(self) -> bool:
        value = self._speed().padding_free
        if value == "auto":
            return True
        return bool(value)

    def _padding_free_forced(self) -> bool:
        return self._speed().padding_free is True

    def _set_attention_implementation(self, model) -> None:
        attn_implementation = getattr(self, "_resolved_attn_implementation", None)
        if attn_implementation is None and self.model_config.attn_implementation not in {None, "auto"}:
            attn_implementation = self.model_config.attn_implementation
        if attn_implementation is None:
            return
        seen: set[int] = set()
        stack = [model]
        while stack:
            current = stack.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            config = getattr(current, "config", None)
            if config is not None:
                setattr(config, "_attn_implementation", attn_implementation)
                setattr(config, "attn_implementation", attn_implementation)
            for attr in ("base_model", "model", "language_model"):
                child = getattr(current, attr, None)
                if child is not None:
                    stack.append(child)

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
                "attn_implementation": getattr(self, "_resolved_attn_implementation", None),
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
        optimizer_cls: Any = torch.optim.AdamW
        optimizer_name = self._speed().optimizer
        if optimizer_name != "adamw_torch" and torch.cuda.is_available():
            try:
                import bitsandbytes as bnb

                optimizer_cls = (
                    bnb.optim.PagedAdamW8bit
                    if optimizer_name == "paged_adamw_8bit"
                    else bnb.optim.AdamW8bit
                )
            except Exception:
                optimizer_cls = torch.optim.AdamW
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=params.learning_rate,
            betas=params.betas,
            eps=params.eps,
            weight_decay=params.weight_decay,
        )
        return self.optimizer

    def _batch_from_data(self, data: list[Datum], *, packed: bool = False) -> _BatchPlan:
        import torch

        if not data:
            raise ValueError("data must not be empty")
        input_ids = [datum.model_input.to_ints() for datum in data]
        lengths = [len(tokens) for tokens in input_ids]
        starts = []
        offset = 0
        for length in lengths:
            starts.append(offset)
            offset += length
        if packed and len(input_ids) > 1:
            flat_tokens = [token for row in input_ids for token in row]
            flat_positions = [pos for row in input_ids for pos, _token in enumerate(row)]
            return _BatchPlan(
                batch={
                    "input_ids": torch.tensor([flat_tokens], dtype=torch.long, device=self._model_device()),
                    "position_ids": torch.tensor([flat_positions], dtype=torch.long, device=self._model_device()),
                    "packed_seq_lengths": torch.tensor(lengths, dtype=torch.int32),
                },
                row_lengths=lengths,
                row_starts=starts,
                packed=True,
            )
        max_len = max(len(tokens) for tokens in input_ids)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
        padded = [tokens + [pad_token_id] * (max_len - len(tokens)) for tokens in input_ids]
        attention = [[1] * len(tokens) + [0] * (max_len - len(tokens)) for tokens in input_ids]
        return _BatchPlan(
            batch={
                "input_ids": torch.tensor(padded, dtype=torch.long, device=self._model_device()),
                "attention_mask": torch.tensor(attention, dtype=torch.long, device=self._model_device()),
            },
            row_lengths=lengths,
            row_starts=[0] * len(lengths),
            packed=False,
        )

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

    def _model_forward(self, batch: dict[str, Any], *, logits_to_keep: int | None = None):
        batch = {key: value for key, value in batch.items() if value is not None}
        if logits_to_keep is None:
            return self.model(**batch)
        keep = max(1, int(logits_to_keep))
        try:
            return self.model(**batch, logits_to_keep=keep)
        except TypeError as first_error:
            try:
                return self.model(**batch, num_logits_to_keep=keep)
            except TypeError:
                if max(int(row.sum().detach().cpu()) for row in batch["attention_mask"]) > 8192:
                    raise RuntimeError(
                        "The model does not accept logits_to_keep/num_logits_to_keep, so this long-context "
                        "loss would materialize full sequence logits and likely OOM."
                    ) from first_error
                return self.model(**batch)

    def _weighted_positions_to_keep(
        self,
        *,
        input_len: int,
        targets: Sequence[int],
        weights: Sequence[float] | None,
        advantages: Sequence[float] | None = None,
    ) -> int:
        positions = []
        for pos, target_token in enumerate(targets[:input_len]):
            if int(target_token) == -100:
                continue
            weight = float(weights[pos]) if weights is not None and pos < len(weights) else 1.0
            advantage = float(advantages[pos]) if advantages is not None and pos < len(advantages) else 1.0
            if weight != 0.0 and advantage != 0.0:
                positions.append(pos)
        if not positions:
            return 1
        return input_len - min(positions)

    def _logit_position(self, logits, *, input_len: int, token_position: int) -> int | None:
        logits_len = int(logits.shape[1])
        logits_start = input_len - logits_len
        local_position = token_position - logits_start
        if local_position < 0 or local_position >= logits_len:
            return None
        return local_position

    def _cross_entropy_loss(self, data: list[Datum]):
        import torch

        plan = self._batch_from_data(data, packed=self._padding_free_requested())
        try:
            return self._cross_entropy_loss_for_plan(data, plan)
        except TypeError:
            if not plan.packed or self._padding_free_forced():
                raise
            return self._cross_entropy_loss_for_plan(data, self._batch_from_data(data, packed=False))

    def _cross_entropy_loss_for_plan(self, data: list[Datum], plan: _BatchPlan):
        import torch

        batch = plan.batch
        keep_counts = []
        prepared_rows = []
        for row_idx, datum in enumerate(data):
            input_len = plan.row_lengths[row_idx]
            has_target_tokens = "target_tokens" in datum.loss_fn_inputs
            raw_targets = datum.loss_fn_inputs.get("target_tokens")
            raw_labels = datum.loss_fn_inputs.get("labels")
            raw_weights = datum.loss_fn_inputs.get("weights")
            if has_target_tokens and raw_targets is not None:
                targets = [int(value) for value in self._loss_tensor_data(raw_targets)][:input_len]
                positions = list(range(min(input_len, len(targets))))
                output_indices = positions
                logit_offset = 0
            else:
                targets = (
                    [int(value) for value in self._loss_tensor_data(raw_labels)][:input_len]
                    if raw_labels is not None
                    else batch["input_ids"][row_idx, :input_len].detach().cpu().tolist()
                )
                positions = list(range(1, min(input_len, len(targets))))
                output_indices = positions
                logit_offset = -1
            weights = None
            if raw_weights is not None:
                weights = [float(value) for value in self._loss_tensor_data(raw_weights)][:input_len]
            logit_positions = [position + logit_offset for position in positions]
            weighted_logit_positions = []
            for position, logit_position in zip(positions, logit_positions):
                if logit_position < 0 or logit_position >= input_len or position >= len(targets):
                    continue
                if int(targets[position]) == -100:
                    continue
                weight = weights[position] if weights is not None and position < len(weights) else 1.0
                if float(weight) != 0.0:
                    weighted_logit_positions.append(logit_position)
            if not plan.packed:
                keep_counts.append(input_len - min(weighted_logit_positions) if weighted_logit_positions else 1)
            prepared_rows.append((input_len, positions, output_indices, logit_positions, targets, weights))

        outputs = self._model_forward(
            batch,
            logits_to_keep=None if plan.packed else (max(keep_counts) if keep_counts else None),
        )
        log_probs = outputs.logits.log_softmax(dim=-1)

        external_logprobs = torch.zeros(
            (len(data), plan.max_len),
            dtype=log_probs.dtype,
            device=log_probs.device,
        )
        model_rows = []
        local_positions = []
        target_tokens = []
        row_indices = []
        output_positions = []
        token_weights = []
        logits_len = int(log_probs.shape[1])

        for row_idx, (input_len, positions, output_indices, logit_positions, targets, weights) in enumerate(prepared_rows):
            for output_index, target_index, logit_index in zip(output_indices, positions, logit_positions):
                target_token = int(targets[target_index])
                if target_token == -100:
                    continue
                weight = 1.0
                if weights is not None:
                    weight = float(weights[target_index])
                if weight == 0.0:
                    continue
                if plan.packed:
                    model_row = 0
                    local_logit_index = plan.row_starts[row_idx] + logit_index
                else:
                    model_row = row_idx
                    local_logit_index = self._logit_position(log_probs, input_len=input_len, token_position=logit_index)
                if local_logit_index is None or local_logit_index < 0 or local_logit_index >= logits_len:
                    continue
                model_rows.append(model_row)
                local_positions.append(local_logit_index)
                target_tokens.append(target_token)
                row_indices.append(row_idx)
                output_positions.append(output_index)
                token_weights.append(weight)

        if model_rows:
            model_rows_t = torch.tensor(model_rows, dtype=torch.long, device=log_probs.device)
            local_positions_t = torch.tensor(local_positions, dtype=torch.long, device=log_probs.device)
            target_tokens_t = torch.tensor(target_tokens, dtype=torch.long, device=log_probs.device)
            gathered = log_probs[model_rows_t, local_positions_t, target_tokens_t]
            row_indices_t = torch.tensor(row_indices, dtype=torch.long, device=log_probs.device)
            output_positions_t = torch.tensor(output_positions, dtype=torch.long, device=log_probs.device)
            weights_t = torch.tensor(token_weights, dtype=log_probs.dtype, device=log_probs.device)
            external_logprobs[row_indices_t, output_positions_t] = gathered
            loss = -(gathered * weights_t).sum()
        else:
            loss = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
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
        plan = self._batch_from_data(data, packed=self._padding_free_requested())
        try:
            return self._policy_loss_for_plan(data, plan, loss_fn=loss_fn, loss_fn_config=loss_fn_config)
        except TypeError:
            if not plan.packed or self._padding_free_forced():
                raise
            return self._policy_loss_for_plan(
                data,
                self._batch_from_data(data, packed=False),
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            )

    def _policy_loss_for_plan(
        self,
        data: list[Datum],
        plan: _BatchPlan,
        *,
        loss_fn: str,
        loss_fn_config: dict[str, Any],
    ):
        import torch

        batch = plan.batch
        prepared_rows = []
        keep_counts = []
        for row_idx, datum in enumerate(data):
            input_len = plan.row_lengths[row_idx]
            targets = [int(value) for value in self._loss_tensor_data(datum.loss_fn_inputs["target_tokens"])][:input_len]
            old_logprobs = [float(value) for value in self._loss_tensor_data(datum.loss_fn_inputs["logprobs"])][:input_len]
            advantages = [float(value) for value in self._loss_tensor_data(datum.loss_fn_inputs["advantages"])][:input_len]
            raw_weights = datum.loss_fn_inputs.get("weights")
            weights = [1.0] * input_len
            if raw_weights is not None:
                weights = [float(value) for value in self._loss_tensor_data(raw_weights)][:input_len]
            if not plan.packed:
                keep_counts.append(
                    self._weighted_positions_to_keep(
                        input_len=input_len,
                        targets=targets,
                        weights=weights,
                        advantages=advantages,
                    )
                )
            prepared_rows.append((input_len, targets, old_logprobs, advantages, weights))

        outputs = self._model_forward(
            batch,
            logits_to_keep=None if plan.packed else (max(keep_counts) if keep_counts else None),
        )
        log_probs = outputs.logits.log_softmax(dim=-1)
        external_logprobs = torch.zeros(
            (len(data), plan.max_len),
            dtype=log_probs.dtype,
            device=log_probs.device,
        )
        external_ratios = torch.zeros_like(external_logprobs)
        clip_low = float(loss_fn_config.get("clip_low_threshold", 0.8))
        clip_high = float(loss_fn_config.get("clip_high_threshold", 1.2))
        model_rows = []
        local_positions = []
        target_tokens = []
        row_indices = []
        output_positions = []
        old_values = []
        weighted_advantages = []
        logits_len = int(log_probs.shape[1])

        for row_idx, (input_len, targets, old_logprobs, advantages, weights) in enumerate(prepared_rows):
            for pos, target_token in enumerate(targets):
                if target_token == -100 or pos >= input_len:
                    continue
                weight = weights[pos] if pos < len(weights) else 1.0
                advantage = advantages[pos] if pos < len(advantages) else 0.0
                if weight == 0.0 or advantage == 0.0:
                    continue
                if plan.packed:
                    model_row = 0
                    local_pos = plan.row_starts[row_idx] + pos
                else:
                    model_row = row_idx
                    local_pos = self._logit_position(log_probs, input_len=input_len, token_position=pos)
                if local_pos is None or local_pos < 0 or local_pos >= logits_len:
                    continue
                model_rows.append(model_row)
                local_positions.append(local_pos)
                target_tokens.append(target_token)
                row_indices.append(row_idx)
                output_positions.append(pos)
                old_values.append(old_logprobs[pos] if pos < len(old_logprobs) else 0.0)
                weighted_advantages.append(weight * advantage)

        if not model_rows:
            total_loss = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
            return total_loss, outputs, external_logprobs, external_ratios

        model_rows_t = torch.tensor(model_rows, dtype=torch.long, device=log_probs.device)
        local_positions_t = torch.tensor(local_positions, dtype=torch.long, device=log_probs.device)
        target_tokens_t = torch.tensor(target_tokens, dtype=torch.long, device=log_probs.device)
        current_logprobs = log_probs[model_rows_t, local_positions_t, target_tokens_t]
        old_logprobs_t = torch.tensor(old_values, dtype=log_probs.dtype, device=log_probs.device)
        ratio = torch.exp(current_logprobs - old_logprobs_t)
        advantages_t = torch.tensor(weighted_advantages, dtype=log_probs.dtype, device=log_probs.device)
        row_indices_t = torch.tensor(row_indices, dtype=torch.long, device=log_probs.device)
        output_positions_t = torch.tensor(output_positions, dtype=torch.long, device=log_probs.device)
        external_logprobs[row_indices_t, output_positions_t] = current_logprobs
        external_ratios[row_indices_t, output_positions_t] = ratio
        if loss_fn == "importance_sampling":
            token_losses = -ratio * advantages_t
        elif loss_fn == "ppo":
            clipped_ratio = ratio.clamp(clip_low, clip_high)
            token_losses = -torch.minimum(ratio * advantages_t, clipped_ratio * advantages_t)
        elif loss_fn == "cispo":
            clipped_ratio = ratio.detach().clamp(clip_low, clip_high)
            token_losses = -clipped_ratio * advantages_t * current_logprobs
        else:
            raise UnsupportedLossError(f"Unsupported loss: {loss_fn}")
        total_loss = token_losses.sum()

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
        plan = self._batch_from_data(data, packed=False)
        outputs = self.model(**plan.batch)
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
        model = self._unwrap_model()
        if hasattr(model, "fast_generate") or hasattr(model, "generate"):
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
        generate = getattr(model, "fast_generate", None)
        if not callable(generate):
            generate = model.generate
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
            "output_scores": sampling_params.logprobs_max_tokens is not None,
            "pad_token_id": pad_token_id,
        }
        if sampling_params.max_time is not None:
            kwargs["max_time"] = float(sampling_params.max_time)
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
            generated = generate(**kwargs)

        sequences = generated.sequences.detach().cpu().tolist()
        outputs = []
        prompt_length = len(prompt_tokens)
        for row_index, sequence in enumerate(sequences):
            completion_tokens = list(sequence[prompt_length:])
            finish_reason = self._finish_reason(completion_tokens, sampling_params.max_tokens)
            if sampling_params.max_time is not None and finish_reason == "stop":
                finish_reason = "time"
            trimmed_tokens, stop_reason = self._trim_at_first_stop(completion_tokens, stop_token_ids)
            if stop_reason is not None:
                finish_reason = stop_reason
            outputs.append((trimmed_tokens, [], finish_reason))
        logprob_rows = None
        if sampling_params.logprobs_max_tokens is not None:
            logprob_rows = self._completion_logprobs_from_generate_scores(
                generated,
                [tokens for tokens, _logprobs, _reason in outputs],
                logprobs_max_tokens=sampling_params.logprobs_max_tokens,
            )
        if logprob_rows is None:
            logprob_rows = self._completion_logprobs_batch(
                prompt_tokens,
                [tokens for tokens, _logprobs, _reason in outputs],
                logprobs_max_tokens=sampling_params.logprobs_max_tokens,
            )
        return [
            (tokens, logprobs, finish_reason)
            for (tokens, _empty_logprobs, finish_reason), logprobs in zip(outputs, logprob_rows)
        ]

    def _completion_logprobs_from_generate_scores(
        self,
        generated: Any,
        completion_rows: list[list[int]],
        *,
        logprobs_max_tokens: int | None = None,
    ) -> list[list[float | None]] | None:
        import torch

        scores = getattr(generated, "scores", None)
        if scores is None:
            return None
        cap = logprobs_max_tokens
        if cap is None:
            cap = len(scores)
        cap = max(0, int(cap))
        rows: list[list[float | None]] = [[] for _completion in completion_rows]
        for token_index, score in enumerate(scores[:cap]):
            log_probs = torch.log_softmax(score.float(), dim=-1)
            for row_index, completion in enumerate(completion_rows):
                if token_index >= len(completion):
                    continue
                token = int(completion[token_index])
                rows[row_index].append(float(log_probs[row_index, token].detach().cpu()))
        return rows

    def _completion_logprobs_batch(
        self,
        prompt_tokens: list[int],
        completion_rows: list[list[int]],
        *,
        logprobs_max_tokens: int | None = None,
    ) -> list[list[float | None]]:
        import torch

        if not completion_rows:
            return []
        if logprobs_max_tokens is not None:
            cap = max(0, int(logprobs_max_tokens))
            completion_rows = [completion[:cap] for completion in completion_rows]
        if all(not completion for completion in completion_rows):
            return [[] for _completion in completion_rows]
        prompt_len = len(prompt_tokens)
        nonempty = [(index, completion) for index, completion in enumerate(completion_rows) if completion]
        max_completion = max(len(completion) for _index, completion in nonempty)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
        full_rows = [prompt_tokens + completion for _index, completion in nonempty]
        max_len = max(len(row) for row in full_rows)
        input_ids = torch.tensor(
            [row + [pad_token_id] * (max_len - len(row)) for row in full_rows],
            dtype=torch.long,
            device=self._model_device(),
        )
        attention_mask = torch.tensor(
            [[1] * len(row) + [0] * (max_len - len(row)) for row in full_rows],
            dtype=torch.long,
            device=self._model_device(),
        )
        with torch.no_grad():
            model_outputs = self._model_forward(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                logits_to_keep=max_completion,
            )
            log_probs = model_outputs.logits.log_softmax(dim=-1)
        rows: list[list[float | None]] = [[] for _completion in completion_rows]
        logits_len = int(log_probs.shape[1])
        for batch_row, (original_index, completion) in enumerate(nonempty):
            input_len = prompt_len + len(completion)
            values: list[float | None] = []
            for token_index, token in enumerate(completion):
                logit_index = prompt_len + token_index - 1
                local_logit_index = self._logit_position(log_probs, input_len=input_len, token_position=logit_index)
                if local_logit_index is None or local_logit_index < 0 or local_logit_index >= logits_len:
                    values.append(None)
                else:
                    values.append(float(log_probs[batch_row, local_logit_index, int(token)].detach().cpu()))
            rows[original_index] = values
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
        import time

        outputs = []
        for _sample_index in range(num_samples):
            context = list(prompt_tokens)
            completion_tokens: list[int] = []
            logprobs: list[float | None] = []
            finish_reason: str | None = None
            deadline = time.monotonic() + float(sampling_params.max_time) if sampling_params.max_time is not None else None

            for _step in range(sampling_params.max_tokens):
                if deadline is not None and time.monotonic() >= deadline:
                    finish_reason = "time"
                    break
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
            if sampling_params.logprobs_max_tokens is not None:
                logprobs = logprobs[: max(0, int(sampling_params.logprobs_max_tokens))]
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

    def _checkpoint_target(self, path: str | None, *, prefix: str) -> Path:
        if path is None:
            return new_checkpoint_path(self.checkpoint_root, prefix, self.step)
        raw = str(path)
        explicit_uri = raw.startswith(("local://", "tinker://"))
        explicit_path = Path(raw).is_absolute() or Path(raw).name != raw
        if explicit_uri or explicit_path:
            return resolve_path(raw)
        return resolve_path(Path(self.checkpoint_root) / self.session_id / raw)

    def save_state(self, path: str | None = None, include_optimizer: bool = False) -> CheckpointRef:
        target = self._checkpoint_target(path, prefix="state")
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
        target = self._checkpoint_target(path, prefix="sampler")
        checkpoint = None
        try:
            if self.is_rank0:
                checkpoint = self._save_weights(target, include_optimizer=False, kind="sampler_weights")
        finally:
            self._barrier()
        if checkpoint is None:
            return None
        return SaveWeightsForSamplerResponse(path=checkpoint.path, checkpoint=checkpoint)
