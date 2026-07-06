"""Fake runtime provider — a GPU-free, dependency-free training backend.

The fake engine is a real (if tiny) model: a trainable bigram logit table
over a 256-token byte vocabulary. ``forward_backward`` computes an exact
cross-entropy over that table, ``optim_step`` applies the accumulated
gradients, so **loss genuinely decreases across steps** and sampling from a
trained table reproduces training text. Checkpoints are manifest-compatible
with the real engine.

Uses: hermetic tests of the full train→sample→eval→checkpoint loop, CI,
`ray-unsloth run --provider fake`, and UI demos — all without a GPU, Ray,
or model downloads.
"""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from ray_unsloth.checkpoints import (
    atomic_checkpoint_dir,
    base_manifest,
    new_checkpoint_path,
    read_manifest,
    resolve_path,
    validate_restore_manifest,
    write_manifest,
)
from ray_unsloth.errors import CheckpointError, UnsupportedLossError
from ray_unsloth.losses import get_loss, validate_datum_inputs
from ray_unsloth.providers.base import (
    LaunchPlan,
    ProviderCapabilities,
    RuntimeProvider,
    SessionProtocol,
)
from ray_unsloth.runtime._resolve import resolve_actor_configs
from ray_unsloth.types import (
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    ForwardOutput,
    GeneratedSequence,
    ModelInput,
    OptimStepResult,
    SampleResponse,
    SamplingParams,
    SaveWeightsForSamplerResponse,
    SaveWeightsResponse,
    TensorData,
    TrainingClientInfo,
)

if TYPE_CHECKING:
    from ray_unsloth.config import RuntimeConfig

VOCAB_SIZE = 256
WEIGHTS_FILE = "fake_weights.json"


class FakeTokenizer:
    """Byte-level tokenizer with the small HF-tokenizer surface examples use."""

    vocab_size = VOCAB_SIZE
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs: Any) -> dict[str, list[int]]:
        del add_special_tokens, kwargs
        return {"input_ids": self.encode(text)}

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        del kwargs
        return [b for b in text.encode("utf-8", errors="replace")]

    def decode(self, tokens: Any, **kwargs: Any) -> str:
        del kwargs
        token_list = list(tokens)
        return bytes(max(0, min(int(t), 255)) for t in token_list).decode("utf-8", errors="replace")

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str | list[int]:
        del kwargs
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in conversation)
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return self.encode(text)
        return text


class _BigramTable:
    """Sparse bigram logit table with exact CE loss and SGD updates."""

    def __init__(self, seed: int):
        self.logits: dict[int, dict[int, float]] = defaultdict(dict)
        self.grads: dict[int, dict[int, float]] = defaultdict(dict)
        self.seed = seed

    def _row(self, prev: int) -> dict[int, float]:
        return self.logits.get(prev, {})

    def _log_softmax(self, prev: int, token: int) -> float:
        row = self._row(prev)
        # Unset entries have logit 0. log-sum-exp over sparse row + zeros.
        max_logit = max(row.values(), default=0.0)
        max_logit = max(max_logit, 0.0)
        sum_exp = (VOCAB_SIZE - len(row)) * math.exp(0.0 - max_logit)
        sum_exp += sum(math.exp(v - max_logit) for v in row.values())
        log_z = max_logit + math.log(sum_exp)
        return row.get(token, 0.0) - log_z

    def _probs_row(self, prev: int) -> list[float]:
        row = self._row(prev)
        logits = [row.get(t, 0.0) for t in range(VOCAB_SIZE)]
        max_logit = max(logits)
        exps = [math.exp(v - max_logit) for v in logits]
        total = sum(exps)
        return [v / total for v in exps]

    def accumulate_ce_grad(self, prev: int, target: int, weight: float) -> float:
        """Cross-entropy gradient for one (prev -> target) transition; returns -logprob."""
        logprob = self._log_softmax(prev, target)
        probs = self._probs_row(prev)
        grad_row = self.grads[prev]
        # d(-logp[target])/d(logit[j]) = p[j] - 1[j == target]
        for token, prob in enumerate(probs):
            if abs(prob) < 1e-9 and token != target:
                continue
            delta = (prob - (1.0 if token == target else 0.0)) * weight
            grad_row[token] = grad_row.get(token, 0.0) + delta
        return -logprob * weight

    def accumulate_scaled_grad(self, prev: int, target: int, scale: float) -> None:
        """Accumulate d(scale * -logprob(target|prev))/dlogits (policy-gradient step)."""
        probs = self._probs_row(prev)
        grad_row = self.grads[prev]
        for token, prob in enumerate(probs):
            if abs(prob) < 1e-9 and token != target:
                continue
            delta = (prob - (1.0 if token == target else 0.0)) * scale
            grad_row[token] = grad_row.get(token, 0.0) + delta

    def apply_grads(self, learning_rate: float) -> None:
        for prev, grad_row in self.grads.items():
            logit_row = self.logits[prev]
            for token, grad in grad_row.items():
                logit_row[token] = logit_row.get(token, 0.0) - learning_rate * grad
        self.grads = defaultdict(dict)

    def logprob(self, prev: int, token: int) -> float:
        return self._log_softmax(prev, token)

    def sample_next(self, prev: int, rng: random.Random, temperature: float, top_p: float) -> tuple[int, float]:
        probs = self._probs_row(prev)
        if temperature <= 0.0:
            token = max(range(VOCAB_SIZE), key=lambda t: probs[t])
            return token, math.log(max(probs[token], 1e-12))
        if temperature != 1.0:
            scaled = [p ** (1.0 / temperature) for p in probs]
            total = sum(scaled)
            probs = [p / total for p in scaled]
        if top_p < 1.0:
            ranked = sorted(range(VOCAB_SIZE), key=lambda t: -probs[t])
            kept, cumulative = [], 0.0
            for token in ranked:
                kept.append(token)
                cumulative += probs[token]
                if cumulative >= top_p:
                    break
            mass = sum(probs[t] for t in kept)
            choice = rng.random() * mass
            for token in kept:
                choice -= probs[token]
                if choice <= 0.0:
                    return token, math.log(max(probs[token], 1e-12))
            return kept[-1], math.log(max(probs[kept[-1]], 1e-12))
        choice = rng.random()
        for token in range(VOCAB_SIZE):
            choice -= probs[token]
            if choice <= 0.0:
                return token, math.log(max(probs[token], 1e-12))
        return VOCAB_SIZE - 1, math.log(max(probs[-1], 1e-12))

    def to_json(self) -> dict[str, Any]:
        return {str(prev): row for prev, row in self.logits.items() if row}

    @classmethod
    def from_json(cls, data: dict[str, Any], seed: int) -> _BigramTable:
        table = cls(seed)
        for prev, row in data.items():
            table.logits[int(prev)] = {int(t): float(v) for t, v in row.items()}
        return table


def _tensor_values(value: Any) -> list[float]:
    if isinstance(value, TensorData):
        data = value.data
        return [float(v) for v in (data.tolist() if hasattr(data, "tolist") else data)]
    if hasattr(value, "tolist"):
        return [float(v) for v in value.tolist()]
    return [float(v) for v in value]


class FakeTrainerActor:
    """GPU-free trainer actor with the full engine method surface."""

    def __init__(
        self,
        *,
        session_id: str,
        base_model: str,
        lora_rank: int,
        checkpoint_root: str,
        seed: int = 3407,
        model_path: str | None = None,
        with_optimizer: bool = False,
        metadata: dict[str, Any] | None = None,
    ):
        self.session_id = session_id
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.checkpoint_root = checkpoint_root
        self.seed = seed
        self.metadata = dict(metadata or {})
        self.step = 0
        self.table = _BigramTable(seed)
        self.tokenizer = FakeTokenizer()
        self._custom_losses: dict[str, Any] = {}
        self._rng = random.Random(seed)
        if model_path is not None:
            if with_optimizer:
                self.load_state_with_optimizer(model_path)
            else:
                self.load_state(model_path)

    # -- info ---------------------------------------------------------------

    def get_tokenizer(self) -> FakeTokenizer:
        return self.tokenizer

    def get_info(self) -> TrainingClientInfo:
        return TrainingClientInfo(
            session_id=self.session_id,
            base_model=self.base_model,
            lora_rank=self.lora_rank,
            step=self.step,
            metadata=dict(self.metadata),
        )

    def get_base_model(self) -> str:
        return self.base_model

    # -- losses ---------------------------------------------------------------

    def register_custom_loss(self, name: str, loss_fn: Any) -> None:
        self._custom_losses[name] = loss_fn

    def _datum_rows(self, data: list[Datum]) -> list[list[int]]:
        return [datum.model_input.to_ints() for datum in data]

    def _supervised_pass(self, data: list[Datum], *, train: bool) -> tuple[float, list[dict[str, Any]]]:
        total_loss, total_weight = 0.0, 0.0
        loss_fn_outputs: list[dict[str, Any]] = []
        for datum in data:
            tokens = datum.model_input.to_ints()
            raw_targets = datum.loss_fn_inputs.get("target_tokens", datum.loss_fn_inputs.get("labels"))
            targets = [int(v) for v in _tensor_values(raw_targets)] if raw_targets is not None else list(tokens)
            raw_weights = datum.loss_fn_inputs.get("weights")
            weights = (
                [float(v) for v in _tensor_values(raw_weights)] if raw_weights is not None else [1.0] * len(targets)
            )
            row_logprobs: list[float] = []
            for pos, target in enumerate(targets[: len(tokens)]):
                prev = tokens[pos - 1] if pos > 0 else 0
                weight = weights[pos] if pos < len(weights) else 1.0
                if target == -100 or weight == 0.0:
                    row_logprobs.append(0.0)
                    continue
                if train:
                    total_loss += self.table.accumulate_ce_grad(prev, int(target), weight)
                else:
                    total_loss += -self.table.logprob(prev, int(target)) * weight
                row_logprobs.append(self.table.logprob(prev, int(target)))
                total_weight += weight
            loss_fn_outputs.append(
                {"logprobs": TensorData(data=row_logprobs, dtype="float32", shape=[len(row_logprobs)])}
            )
        del total_weight  # engine reports summed loss; parity kept
        return total_loss, loss_fn_outputs

    def _policy_pass(
        self,
        data: list[Datum],
        *,
        loss_fn: str,
        loss_fn_config: dict[str, Any] | None,
        train: bool,
    ) -> tuple[float, list[dict[str, Any]]]:
        spec = get_loss(loss_fn)
        if spec.kind != "policy_gradient" or spec.token_loss is None:
            raise UnsupportedLossError(f"Unsupported loss '{loss_fn}'.")
        config = spec.merged_config(loss_fn_config)
        total_loss = 0.0
        loss_fn_outputs: list[dict[str, Any]] = []
        for index, datum in enumerate(data):
            validate_datum_inputs(spec, datum.loss_fn_inputs, datum_index=index)
            tokens = datum.model_input.to_ints()
            targets = [int(v) for v in _tensor_values(datum.loss_fn_inputs["target_tokens"])]
            old_logprobs = _tensor_values(datum.loss_fn_inputs["logprobs"])
            advantages = _tensor_values(datum.loss_fn_inputs["advantages"])
            raw_weights = datum.loss_fn_inputs.get("weights")
            weights = _tensor_values(raw_weights) if raw_weights is not None else [1.0] * len(targets)
            row_logprobs = [0.0] * len(targets)
            row_ratios = [0.0] * len(targets)
            for pos, target in enumerate(targets[: len(tokens)]):
                weight = weights[pos] if pos < len(weights) else 1.0
                advantage = advantages[pos] if pos < len(advantages) else 0.0
                if target == -100 or weight == 0.0 or advantage == 0.0:
                    continue
                prev = tokens[pos - 1] if pos > 0 else 0
                current = self.table.logprob(prev, int(target))
                old = old_logprobs[pos] if pos < len(old_logprobs) else 0.0
                ratio = math.exp(current - old)
                row_logprobs[pos] = current
                row_ratios[pos] = ratio
                token_loss = float(
                    spec.token_loss(
                        ratio=_Scalar(ratio),
                        advantages=_Scalar(weight * advantage),
                        current_logprobs=_Scalar(current),
                        config=config,
                    )
                )
                total_loss += token_loss
                if train:
                    # d(token_loss)/d(current_logprob) via finite structure:
                    # for -ratio*A: dL/dlogp = -ratio*A. Generic: numeric derivative.
                    epsilon = 1e-4
                    bumped = float(
                        spec.token_loss(
                            ratio=_Scalar(math.exp(current + epsilon - old)),
                            advantages=_Scalar(weight * advantage),
                            current_logprobs=_Scalar(current + epsilon),
                            config=config,
                        )
                    )
                    dloss_dlogp = (bumped - token_loss) / epsilon
                    self.table.accumulate_scaled_grad(prev, int(target), dloss_dlogp)
            loss_fn_outputs.append(
                {
                    "logprobs": TensorData(data=row_logprobs, dtype="float32", shape=[len(row_logprobs)]),
                    "ratios": TensorData(data=row_ratios, dtype="float32", shape=[len(row_ratios)]),
                }
            )
        return total_loss, loss_fn_outputs

    def forward(self, data: list[Datum], loss_fn: str = "cross_entropy") -> ForwardOutput:
        spec = get_loss(loss_fn)
        if spec.kind == "supervised":
            loss, outputs = self._supervised_pass(data, train=False)
        else:
            loss, outputs = self._policy_pass(data, loss_fn=loss_fn, loss_fn_config=None, train=False)
        return ForwardOutput(loss=loss, metrics={"loss": loss, "loss:sum": loss}, loss_fn_outputs=outputs)

    def forward_backward(
        self,
        data: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, Any] | None = None,
    ) -> ForwardBackwardOutput:
        spec = get_loss(loss_fn)
        if spec.kind == "supervised":
            loss, outputs = self._supervised_pass(data, train=True)
        else:
            loss, outputs = self._policy_pass(data, loss_fn=loss_fn, loss_fn_config=loss_fn_config, train=True)
        if not math.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss}")
        return ForwardBackwardOutput(loss=loss, metrics={"loss": loss, "loss:sum": loss}, loss_fn_outputs=outputs)

    def forward_backward_custom(
        self,
        data: list[Datum],
        loss_fn: Any,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> ForwardBackwardOutput:
        if isinstance(loss_fn, str):
            if loss_fn not in self._custom_losses:
                raise UnsupportedLossError(f"Custom loss is not registered: {loss_fn}")
            loss_fn = self._custom_losses[loss_fn]
        outputs = _FakeModelOutputs(self.table, self._datum_rows(data))
        loss, metrics = loss_fn(outputs, data, loss_fn_config or {})
        value = float(loss)
        # Custom losses can't backprop through the fake table; treat the loss
        # value as a supervised CE nudge on the datum tokens so training moves.
        for datum in data:
            tokens = datum.model_input.to_ints()
            for pos in range(1, len(tokens)):
                self.table.accumulate_ce_grad(tokens[pos - 1], tokens[pos], 1e-3)
        return ForwardBackwardOutput(
            loss=value,
            metrics={"loss": value, **{k: float(v) for k, v in metrics.items()}},
        )

    def optim_step(self, adam_params: AdamParams) -> OptimStepResult:
        self.table.apply_grads(adam_params.learning_rate)
        self.step += 1
        return OptimStepResult(step=self.step, metrics={"step": float(self.step)})

    # -- sampling -------------------------------------------------------------

    def compute_logprobs(self, prompt: ModelInput | list[int]) -> list[float | None]:
        tokens = prompt.to_ints() if isinstance(prompt, ModelInput) else list(prompt)
        if len(tokens) < 2:
            return [None] * len(tokens)
        return [None] + [self.table.logprob(tokens[i - 1], tokens[i]) for i in range(1, len(tokens))]

    def sample(
        self,
        prompt: ModelInput | list[int],
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> SampleResponse:
        del topk_prompt_logprobs
        sampling_params = sampling_params or SamplingParams()
        prompt_tokens = prompt.to_ints() if isinstance(prompt, ModelInput) else list(prompt)
        rng = random.Random(sampling_params.seed if sampling_params.seed is not None else self._rng.random())
        sequences = []
        for _ in range(num_samples):
            generated: list[int] = []
            logprobs: list[float] = []
            prev = prompt_tokens[-1] if prompt_tokens else 0
            finish_reason = "length"
            for _ in range(sampling_params.max_tokens):
                token, logprob = self.table.sample_next(prev, rng, sampling_params.temperature, sampling_params.top_p)
                generated.append(token)
                logprobs.append(logprob)
                prev = token
                if token == self.tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break
                text_so_far = self.tokenizer.decode(generated)
                if any(stop in text_so_far for stop in sampling_params.stop):
                    finish_reason = "stop"
                    break
            sequences.append(
                GeneratedSequence(
                    tokens=generated,
                    text=self.tokenizer.decode(generated),
                    logprobs=logprobs,
                    finish_reason=finish_reason,
                )
            )
        prompt_logprobs = self.compute_logprobs(prompt_tokens) if include_prompt_logprobs else None
        return SampleResponse(sequences=sequences, prompt_logprobs=prompt_logprobs)

    # -- checkpoints ------------------------------------------------------------

    def _checkpoint_target(self, path: str | None, *, kind: str) -> Any:
        from pathlib import Path

        if path is None:
            return new_checkpoint_path(self.checkpoint_root, f"fake-{kind}", self.step)
        raw = str(path)
        explicit_uri = raw.startswith(("local://", "tinker://"))
        explicit_path = Path(raw).is_absolute() or Path(raw).name != raw
        if explicit_uri or explicit_path:
            return resolve_path(raw)
        return resolve_path(Path(self.checkpoint_root) / self.session_id / raw)

    def _write_checkpoint(self, path: str | None, *, kind: str, has_optimizer: bool) -> str:
        target = self._checkpoint_target(path, kind=kind)
        with atomic_checkpoint_dir(target) as tmp_dir:
            (tmp_dir / WEIGHTS_FILE).write_text(json.dumps(self.table.to_json()))
        manifest = base_manifest(
            kind=kind,
            step=self.step,
            base_model=self.base_model,
            lora={"rank": self.lora_rank, "target_modules": None},
            has_optimizer=has_optimizer,
            extra={"engine": "fake", "session_id": self.session_id, "seed": self.seed},
        )
        write_manifest(target, manifest)
        return str(target)

    def save_state(self, path: str | None = None) -> SaveWeightsResponse:
        from ray_unsloth.checkpoints import checkpoint_ref

        saved = self._write_checkpoint(path, kind="training_state", has_optimizer=False)
        return checkpoint_ref(saved, False)

    def save_state_with_optimizer(self, path: str | None = None) -> SaveWeightsResponse:
        from ray_unsloth.checkpoints import checkpoint_ref

        saved = self._write_checkpoint(path, kind="training_state", has_optimizer=True)
        return checkpoint_ref(saved, True)

    def _load_checkpoint(self, path: str) -> None:
        manifest = read_manifest(path)
        validate_restore_manifest(manifest, path=path, base_model=self.base_model, lora_rank=self.lora_rank)
        weights_path = resolve_path(path) / WEIGHTS_FILE
        if not weights_path.exists():
            raise CheckpointError(f"Checkpoint at {path} has no {WEIGHTS_FILE}; it was not saved by the fake engine.")
        self.table = _BigramTable.from_json(json.loads(weights_path.read_text()), self.seed)
        step = manifest.get("step")
        if isinstance(step, int):
            self.step = step

    def load_state(self, path: str) -> None:
        # Match the real engine: restore is side-effect-only and returns None.
        self._load_checkpoint(path)

    def load_state_with_optimizer(self, path: str) -> None:
        self.load_state(path)

    def save_weights_for_sampler(self, path: str | None = None) -> SaveWeightsForSamplerResponse:
        saved = self._write_checkpoint(path, kind="sampler", has_optimizer=False)
        from ray_unsloth.checkpoints import checkpoint_ref

        return SaveWeightsForSamplerResponse(path=saved, checkpoint=checkpoint_ref(saved, False))

    def save_sampler_with_download_url(self, path: str | None = None, ttl_seconds: int = 3600) -> Any:
        from ray_unsloth.download import (
            archive_relpath,
            load_or_create_secret,
            make_token,
            pack_lora_archive,
        )
        from ray_unsloth.types import SamplerDownloadResponse

        save = self.save_weights_for_sampler(path)
        archive = pack_lora_archive(save.path)
        secret = load_or_create_secret(self.checkpoint_root)
        relpath = archive_relpath(archive, self.checkpoint_root)
        expires_at = int(time.time()) + max(60, int(ttl_seconds))
        return SamplerDownloadResponse(
            path=save.path,
            archive_path=str(archive),
            archive_relpath=relpath,
            token=make_token(relpath, expires_at, secret),
            expires_at=expires_at,
            url=None,
            checkpoint=save.checkpoint,
        )


class _Scalar(float):
    """Float with the tiny tensor surface the built-in token losses use."""

    def clamp(self, low: float, high: float) -> _Scalar:
        return _Scalar(min(max(float(self), low), high))

    def detach(self) -> _Scalar:
        return self


class _FakeModelOutputs:
    """Duck-typed `outputs` object handed to custom losses by the fake engine."""

    def __init__(self, table: _BigramTable, rows: list[list[int]]):
        self.table = table
        self.rows = rows
        self.logits = None  # custom losses that need real logits require the GPU engine

    def sequence_logprob(self, row: int) -> float:
        tokens = self.rows[row]
        return sum(self.table.logprob(tokens[i - 1], tokens[i]) for i in range(1, len(tokens)))


class FakeSamplerActor:
    """Sampler-only view over a fake trainer checkpoint (or a fresh table)."""

    def __init__(
        self,
        *,
        session_id: str,
        base_model: str,
        checkpoint_root: str,
        lora_rank: int,
        model_path: str | None = None,
        seed: int = 3407,
    ):
        self._trainer = FakeTrainerActor(
            session_id=session_id,
            base_model=base_model,
            lora_rank=lora_rank,
            checkpoint_root=checkpoint_root,
            seed=seed,
        )
        if model_path is not None:
            self._trainer._load_checkpoint(model_path)

    def sample(self, *args: Any, **kwargs: Any) -> SampleResponse:
        return self._trainer.sample(*args, **kwargs)

    def compute_logprobs(self, prompt: ModelInput | list[int]) -> list[float | None]:
        return self._trainer.compute_logprobs(prompt)

    def get_tokenizer(self) -> FakeTokenizer:
        return self._trainer.get_tokenizer()

    def get_base_model(self) -> str:
        return self._trainer.get_base_model()


class FakeSession:
    """In-process session satisfying SessionProtocol with plain-object actors."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.training_actors: dict[str, Any] = {}
        self.sampler_actors: dict[str, list[Any]] = {}

    def create_training_actor(
        self,
        *,
        base_model: str | None = None,
        lora_rank: int | None = None,
        seed: int | None = None,
        target_modules: list[str] | None = None,
        model_path: str | None = None,
        with_optimizer: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, Any]:
        session_id = f"train-{uuid.uuid4().hex}"
        model_config, lora_config = resolve_actor_configs(
            self.config,
            base_model=base_model,
            lora_rank=lora_rank,
            seed=seed,
            target_modules=target_modules,
        )
        actor = FakeTrainerActor(
            session_id=session_id,
            base_model=model_config.base_model,
            lora_rank=lora_config.rank,
            checkpoint_root=self.config.checkpoint_root,
            seed=lora_config.random_state,
            model_path=model_path,
            with_optimizer=with_optimizer,
            metadata=metadata,
        )
        self.training_actors[session_id] = actor
        return session_id, actor

    def create_sampler_actors(
        self,
        *,
        base_model: str | None = None,
        model_path: str | None = None,
        replicas: int | None = None,
    ) -> tuple[str, list[Any]]:
        session_id = f"sample-{uuid.uuid4().hex}"
        model_config, lora_config = resolve_actor_configs(self.config, base_model=base_model)
        count = replicas if replicas is not None else self.config.resources.sampler_replicas
        actors = [
            FakeSamplerActor(
                session_id=session_id,
                base_model=model_config.base_model,
                checkpoint_root=self.config.checkpoint_root,
                lora_rank=lora_config.rank,
                model_path=model_path,
            )
            for _ in range(max(1, count))
        ]
        self.sampler_actors[session_id] = actors
        return session_id, actors

    def close(self) -> None:
        self.training_actors.clear()
        self.sampler_actors.clear()


class FakeProvider(RuntimeProvider):
    name = "fake"
    description = "In-process GPU-free engine (trainable bigram model) for tests, CI, and demos."

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            description=self.description,
            kind="execution",
            multi_node=False,
            live_policy_sampling=True,
            gpu_types=["none required"],
            cost_estimation=False,
        )

    def plan(self, config: RuntimeConfig) -> LaunchPlan:
        return LaunchPlan(
            provider=self.name,
            summary="Run the in-process fake engine (no GPU, no Ray, no downloads).",
            steps=[
                f"Create in-process fake trainer for '{config.model.base_model}' "
                f"(byte-level bigram model, seed {config.lora.random_state})",
                f"Checkpoints under {config.checkpoint_root}/",
            ],
        )

    def connect(self, config: RuntimeConfig) -> SessionProtocol:
        return FakeSession(config)
