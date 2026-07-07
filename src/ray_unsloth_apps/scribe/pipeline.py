"""Scribe pipeline stages and orchestration."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient
from ray_unsloth.config import RuntimeConfig, load_config
from ray_unsloth.export import ExportReport, export_checkpoint
from ray_unsloth.recipes import GrpoConfig, PromptSpec, Rubric, RubricTerm, grpo_round, sft_epoch, text_completion_datum
from ray_unsloth.recipes.renderers import get_renderer
from ray_unsloth.store import RunStore
from ray_unsloth_apps.scribe.classifier import auc as classifier_auc
from ray_unsloth_apps.scribe.classifier import train_classifier
from ray_unsloth_apps.scribe.ingest import Passage, ingest_paths, load_passages, save_passages
from ray_unsloth_apps.scribe.profile import (
    StyleProfile,
    build_profile,
    capability_report,
    copy_overlap,
    load_profile,
    save_profile,
    stylometry_distance,
)
from ray_unsloth_apps.scribe.prompts import (
    NEUTRAL_PARAGRAPHS,
    backtranslate_prompts,
    load_prompts,
    save_prompts,
    task_bank,
)


@dataclass(slots=True)
class ScribeConfig:
    sft_epochs: int = 2
    sft_batch_size: int = 4
    rl_rounds: int = 2
    group_size: int = 6
    batches_per_round: int = 3
    max_tokens: int = 96
    max_prompts: int = 24
    eval_generations: int = 40
    seed: int = 0
    # Engine-appropriate defaults; the fake provider's tests override these
    # (its table SGD needs far larger steps than real LoRA training).
    sft_learning_rate: float = 2e-4
    rl_learning_rate: float = 1e-5
    run_name: str = "scribe"
    fluency_fn: Callable[[Any, str], float] | None = None

    @classmethod
    def from_yaml(cls, config: str | Path | RuntimeConfig | dict[str, Any] | None) -> ScribeConfig:
        runtime = load_config(config)
        return cls.from_dict(runtime.scribe)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ScribeConfig:
        data = dict(data or {})
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


def stage_sft(service: Any, passages: list[Passage], prompts: list[dict[str, Any]], config: ScribeConfig) -> str:
    return _stage_sft_details(service, passages, prompts, config)["checkpoint"]


def stage_rl(
    service: Any, sft_checkpoint: str, prompts: list[dict[str, Any]], profile: StyleProfile, config: ScribeConfig
) -> str:
    return _stage_rl_details(service, sft_checkpoint, prompts, profile, config)["checkpoint"]


def stage_eval(
    service: Any, checkpoint: str, passages: list[Passage], profile: StyleProfile, config: ScribeConfig
) -> dict[str, Any]:
    store = RunStore(_store_root(service))
    heldout, train = _split_passages(passages, seed=config.seed)
    if not heldout:
        heldout = list(passages[:1])
    sampling_client = service.create_sampling_client(model_path=checkpoint)
    bank_prompt_specs = _bank_prompt_specs(
        [{"text": text, "source": "bank"} for text in task_bank()], sampling_client.get_tokenizer().result()
    )
    generations = _generate_samples(
        sampling_client,
        bank_prompt_specs,
        max_samples=config.eval_generations,
        max_tokens=config.max_tokens,
        seed=config.seed,
    )
    report = {
        "checkpoint": checkpoint,
        "app": "scribe",
        "stage": "eval",
        "auc": classifier_auc([passage.text for passage in heldout], generations),
        "stylometry_mean": _mean(stylometry_distance(text, profile) for text in generations),
        "fluency_mean": _mean(_fluency_score(sampling_client, text) for text in generations),
        "n": len(generations),
        "heldout_n": len(heldout),
        "train_n": len(train),
    }
    store.record_eval(report)
    return report


def materialize_checkpoint(checkpoint: str, runtime: RuntimeConfig, workdir: str | Path) -> str:
    """Return a locally readable copy of ``checkpoint``.

    Exporters read manifests and weights from the local filesystem. When the
    checkpoint lives on a remote provider volume (Modal's ``/checkpoints``
    mount), pull it down with ``modal volume get`` into the workdir first.
    """
    import subprocess

    from ray_unsloth.checkpoints import resolve_path
    from ray_unsloth.errors import CheckpointError

    if resolve_path(checkpoint).exists():
        return checkpoint
    mount = runtime.modal.volume_mount_path.rstrip("/")
    raw = str(checkpoint)
    if raw.startswith(mount + "/"):
        relpath = raw[len(mount) + 1 :]
    elif raw.startswith("/__modal/volumes/"):
        # The engine records the container-resolved volume path
        # (/__modal/volumes/<volume-id>/<relpath>), not the mount alias.
        relpath = raw.removeprefix("/__modal/volumes/").split("/", 1)[1]
    else:
        raise CheckpointError(
            f"Checkpoint is not readable locally and is not under the Modal volume mount: {raw}",
            code="RU-2001",
            hint="Download the checkpoint to this machine before exporting.",
        )
    dest = Path(workdir).expanduser() / "export-src" / relpath
    if not (dest / "manifest.json").exists():
        import shutil

        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # `modal volume get` uses cp -r semantics for directories: the local
        # destination is the PARENT, and the directory is created inside it.
        result = subprocess.run(
            ["modal", "volume", "get", "--force", runtime.modal.volume_name, relpath, str(dest.parent)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not (dest / "manifest.json").exists():
            detail = result.stderr.strip()[-500:] or "download completed but manifest.json is missing"
            raise CheckpointError(
                f"Could not download checkpoint '{relpath}' from Modal volume '{runtime.modal.volume_name}': {detail}",
                code="RU-2001",
                hint="Check `modal profile current` and the volume name in your config.",
            )
    return str(dest)


def stage_export(
    checkpoint: str,
    target: str = "ollama",
    *,
    runtime: RuntimeConfig | None = None,
    workdir: str | Path = "scribe",
) -> ExportReport:
    if runtime is not None:
        checkpoint = materialize_checkpoint(checkpoint, runtime, workdir)
    return export_checkpoint(target, checkpoint)


def run_pipeline(
    config_path: str | Path | RuntimeConfig | dict[str, Any] | None,
    workdir: str | Path,
    *,
    steps: list[str] | None,
    paths: list[str] | None = None,
) -> dict[str, Any]:
    runtime = load_config(config_path)
    config = ScribeConfig.from_yaml(config_path)
    workdir_path = Path(workdir).expanduser()
    workdir_path.mkdir(parents=True, exist_ok=True)
    corpus_path = workdir_path / "corpus" / "passages.jsonl"
    profile_path = workdir_path / "profile.json"
    prompts_path = workdir_path / "prompts.jsonl"
    store = RunStore(runtime.store_root)
    service = ServiceClient(config=runtime)
    summary: dict[str, Any] = {"workdir": str(workdir_path), "store_root": str(runtime.store_root)}
    try:
        passages = _load_or_ingest_passages(corpus_path, paths or [])
        summary["ingest"] = {"n_passages": len(passages), "path": str(corpus_path)}

        profile = _load_or_build_profile(profile_path, passages)
        summary["profile"] = {
            "path": str(profile_path),
            "n_passages": profile.n_passages,
            "n_words": profile.n_words,
            "capability": capability_report(profile),
        }

        prompts = _load_or_build_prompts(prompts_path, service, passages, config)
        summary["prompts"] = {"path": str(prompts_path), "n_prompts": len(prompts)}

        checkpoint = _latest_checkpoint_for_stage(store, "rl") or _latest_checkpoint_for_stage(store, "sft")

        if _should_run(steps, "sft"):
            sft_run = _latest_run_for_stage(store, "sft")
            if sft_run is None:
                sft_report = _stage_sft_details(service, passages, prompts, config)
                summary["sft"] = sft_report
                checkpoint = sft_report["checkpoint"]
            else:
                checkpoint = _latest_checkpoint_for_run(store, sft_run.id) or checkpoint
                summary["sft"] = _run_summary(store, sft_run)

        if _should_run(steps, "rl"):
            rl_run = _latest_run_for_stage(store, "rl")
            max_existing_round = _max_round(store)
            if rl_run is None or max_existing_round < config.rl_rounds:
                start_checkpoint = checkpoint
                if start_checkpoint is None:
                    raise ValueError("Scribe RL requires an SFT checkpoint.")
                rl_report = _stage_rl_details(service, start_checkpoint, prompts, profile, config)
                summary["rl"] = rl_report
                checkpoint = rl_report["checkpoint"]
            else:
                checkpoint = _latest_checkpoint_for_run(store, rl_run.id) or checkpoint
                summary["rl"] = _run_summary(store, rl_run)

        if _should_run(steps, "eval") and checkpoint is not None:
            summary["eval"] = stage_eval(service, checkpoint, passages, profile, config)

        if _should_run(steps, "export") and checkpoint is not None:
            export_report = stage_export(checkpoint, runtime=runtime, workdir=workdir_path)
            summary["export"] = export_report.to_dict()

        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    finally:
        service.close()


def _stage_sft_details(
    service: Any, passages: list[Passage], prompts: list[dict[str, Any]], config: ScribeConfig
) -> dict[str, Any]:
    store = RunStore(_store_root(service))
    metadata = {"app": "scribe", "stage": "sft", "round": 0}
    with _managed_training_client(service, name=f"{config.run_name}-sft", metadata=metadata) as training:
        tokenizer = training.get_tokenizer().result()
        datums = _build_sft_datums(passages, prompts, config, tokenizer=tokenizer)
        if not datums:
            raise ValueError("Scribe SFT requires at least one datum.")
        rng = random.Random(config.seed)
        shuffled = list(datums)
        rng.shuffle(shuffled)
        holdout_size = max(1, len(shuffled) // 10)
        heldout = shuffled[:holdout_size]
        train = shuffled[holdout_size:] or shuffled

        before_loss = float(training.forward(heldout).result().loss)
        best_loss = before_loss
        first_checkpoint = training.save_state("scribe-sft").result().path
        best_checkpoint = first_checkpoint
        for epoch in range(max(1, config.sft_epochs)):
            train_losses = sft_epoch(
                training,
                train,
                batch_size=max(1, min(config.sft_batch_size, len(train))),
                adam_params=AdamParams(learning_rate=config.sft_learning_rate),
                shuffle_seed=config.seed + epoch,
            )
            after_loss = float(training.forward(heldout).result().loss)
            if training.run_id is not None:
                store.append_metrics(
                    training.run_id,
                    {
                        "event": "scribe_sft_epoch",
                        "epoch": epoch + 1,
                        "train_loss": train_losses[-1] if train_losses else None,
                        "heldout_loss": after_loss,
                    },
                )
            if after_loss > best_loss:
                break
            best_loss = after_loss
            best_checkpoint = training.save_state("scribe-sft").result().path
        return {
            "checkpoint": best_checkpoint,
            "run_id": training.run_id,
            "before_loss": before_loss,
            "after_loss": best_loss,
            "loss_decreased": best_loss <= before_loss,
            "n_datums": len(datums),
            "n_train": len(train),
            "n_holdout": len(heldout),
        }


def _stage_rl_details(
    service: Any,
    sft_checkpoint: str,
    prompts: list[dict[str, Any]],
    profile: StyleProfile,
    config: ScribeConfig,
) -> dict[str, Any]:
    store = RunStore(_store_root(service))
    prompt_tokenizer = service.create_sampling_client(model_path=sft_checkpoint).get_tokenizer().result()
    prompt_specs = _bank_prompt_specs(prompts, prompt_tokenizer)
    if not prompt_specs:
        raise ValueError("Scribe RL requires at least one bank prompt.")

    checkpoint = sft_checkpoint
    rounds: list[dict[str, Any]] = []
    corpus_passages = [Passage(text=text, source="corpus", kind="text") for text in _positive_texts(prompts)]
    baseline = stage_eval(service, checkpoint, corpus_passages, profile, config)
    previous_auc = float(baseline["auc"])

    for round_no in range(1, config.rl_rounds + 1):
        metadata = {"app": "scribe", "stage": "rl", "round": round_no}
        with _managed_training_client(service, checkpoint=checkpoint, metadata=metadata) as training:
            if training.run_id is not None and hasattr(service, "_store") and service._store is not None:
                service._store.update_run(training.run_id, name=f"{config.run_name}-rl-{round_no}")
            positives = _positive_texts(prompts)
            anchor_datums = _anchor_datums(training, config.seed + round_no, size=16, positives=positives)
            rubric = _build_rubric(training, positives, prompt_specs, profile, config, round_no)
            report = grpo_round(
                training,
                prompt_specs,
                rubric,
                GrpoConfig(
                    group_size=config.group_size,
                    prompts_per_batch=4,
                    batches_per_round=config.batches_per_round,
                    inner_epochs=1,
                    loss_fn="importance_sampling",
                    learning_rate=config.rl_learning_rate,
                    max_tokens=config.max_tokens,
                    seed=config.seed + round_no,
                ),
                anchor_datums=anchor_datums,
            )
            if training.run_id is not None:
                store.append_metrics(
                    training.run_id,
                    {
                        "event": "scribe_rl_round",
                        "round": round_no,
                        "mean_reward": report.mean_reward,
                        "n_datums": report.n_datums,
                        "n_scored_samples": report.n_scored_samples,
                        **{f"term/{name}": value for name, value in report.per_term_means.items()},
                    },
                )
            checkpoint = training.save_state(f"scribe-rl-round{round_no}").result().path
        eval_report = stage_eval(service, checkpoint, corpus_passages, profile, config)
        round_summary = {**asdict(report), "checkpoint": checkpoint, "eval_auc": eval_report["auc"]}
        rounds.append(round_summary)
        if float(eval_report["auc"]) > previous_auc + 0.02:
            checkpoint = rounds[-2]["checkpoint"] if len(rounds) > 1 else sft_checkpoint
            break
        previous_auc = float(eval_report["auc"])

    return {"checkpoint": checkpoint, "baseline_auc": baseline["auc"], "rounds": rounds}


def _build_sft_datums(
    passages: list[Passage],
    prompts: list[dict[str, Any]],
    config: ScribeConfig,
    *,
    tokenizer: Any | None = None,
) -> list[Datum]:
    if not passages:
        return []
    # The MODEL'S tokenizer, always. Building datums with the byte tokenizer
    # only "works" on the fake engine — on a real model it trains on garbage
    # token ids (the run-2 postmortem bug).
    tokenizer = tokenizer if tokenizer is not None else _byte_tokenizer()
    prompt_texts = [str(row.get("text", "")).strip() for row in prompts if str(row.get("text", "")).strip()]
    if not prompt_texts:
        prompt_texts = [passage.text for passage in passages]
    split = max(1, round(len(passages) * 0.7))
    datums: list[Datum] = []
    for index, passage in enumerate(passages[:split]):
        prompt = prompt_texts[index % len(prompt_texts)]
        datums.append(text_completion_datum(tokenizer, prompt.rstrip() + "\n\n", passage.text))
    for passage in passages[split:]:
        tokens = _encode_prompt(tokenizer, passage.text)
        datums.append(Datum(model_input=ModelInput.from_ints(tokens), loss_fn_inputs={"labels": tokens}))
    rng = random.Random(config.seed)
    rng.shuffle(datums)
    return datums


def _load_or_ingest_passages(path: Path, input_paths: list[str]) -> list[Passage]:
    if path.exists():
        return load_passages(path)
    if not input_paths:
        raise ValueError("Scribe ingest requires at least one input path when no corpus exists.")
    passages = ingest_paths(input_paths)
    save_passages(passages, path)
    return passages


def _load_or_build_profile(path: Path, passages: list[Passage]) -> StyleProfile:
    if path.exists():
        return load_profile(path)
    profile = build_profile(passages)
    save_profile(profile, path)
    return profile


def _load_or_build_prompts(
    path: Path, service: Any, passages: list[Passage], config: ScribeConfig
) -> list[dict[str, Any]]:
    if path.exists():
        return load_prompts(path)
    sampler = service.create_sampling_client(base_model=_runtime_model_base(service))
    prompts = backtranslate_prompts(sampler, passages, get_renderer("plain"), max_prompts=config.max_prompts)
    rows = [*prompts, *({"text": text, "source": "bank"} for text in task_bank())]
    save_prompts(rows, path)
    return rows


def _build_rubric(
    training: Any,
    positive_texts: list[str],
    prompt_specs: list[PromptSpec],
    profile: StyleProfile,
    config: ScribeConfig,
    round_no: int,
) -> Rubric:
    live_policy = training.create_live_sampling_client()
    negatives = _generate_samples(
        live_policy,
        prompt_specs,
        max_samples=64,
        max_tokens=config.max_tokens,
        seed=config.seed + round_no,
    )
    negatives.extend(NEUTRAL_PARAGRAPHS)
    classifier = train_classifier(positive_texts, negatives, epochs=200, lr=0.1, seed=config.seed + round_no)

    def style_clf(*, completion_text: str, **_: Any) -> float:
        return 2.0 * classifier.predict_proba(completion_text) - 1.0

    def stylometry(*, completion_text: str, **_: Any) -> float:
        return 1.0 - stylometry_distance(completion_text, profile) / 3.0

    def fluency(*, completion_text: str, **_: Any) -> float:
        if config.fluency_fn is not None:
            return float(config.fluency_fn(live_policy, completion_text))
        return _fluency_score(live_policy, completion_text)

    def task(*, prompt: str, completion_text: str, **_: Any) -> float:
        requested = _parse_requested_words(prompt)
        words = len(completion_text.split())
        low = max(1, int(requested * 0.4))
        high = max(low, int(requested * 1.6))
        banned = any(phrase in completion_text.lower() for phrase in ("as an ai", "here is", "here's a"))
        return 0.0 if banned else (1.0 if low <= words <= high else 0.0)

    def anti_copy(*, completion_text: str, **_: Any) -> float:
        return 1.0 - copy_overlap(completion_text, profile)

    return Rubric(
        terms=[
            RubricTerm(name="style_clf", fn=style_clf, weight=0.35, z_normalize=False),
            RubricTerm(name="stylometry", fn=stylometry, weight=0.25, z_normalize=False),
            RubricTerm(name="fluency", fn=fluency, weight=0.15, z_normalize=False),
            RubricTerm(name="task", fn=task, weight=0.15, z_normalize=False),
            RubricTerm(name="anti_copy", fn=anti_copy, weight=0.10, z_normalize=False, override_below=0.5),
        ]
    )


def _bank_prompt_specs(prompts: list[dict[str, Any]], tokenizer: Any) -> list[PromptSpec]:
    rows = [row for row in prompts if str(row.get("text", "")).strip()]
    specs: list[PromptSpec] = []
    for row in rows:
        if row.get("source") != "bank":
            continue
        text = str(row["text"]).strip()
        specs.append(
            PromptSpec(prompt_text=text, prompt_tokens=_encode_prompt(tokenizer, text), context={"source": "bank"})
        )
    if specs:
        return specs
    for row in rows:
        text = str(row["text"]).strip()
        specs.append(
            PromptSpec(
                prompt_text=text,
                prompt_tokens=_encode_prompt(tokenizer, text),
                context={"source": str(row.get("source", "bank"))},
            )
        )
    return specs


def _positive_texts(prompts: list[dict[str, Any]]) -> list[str]:
    texts = [str(row.get("passage_text", "")).strip() for row in prompts if row.get("source") == "backtranslated"]
    return texts or [str(row.get("text", "")).strip() for row in prompts if str(row.get("text", "")).strip()]


def _generate_samples(
    sampling_client: Any,
    prompts: list[PromptSpec],
    *,
    max_samples: int,
    max_tokens: int,
    seed: int,
) -> list[str]:
    if not prompts:
        return []
    rng = random.Random(seed)
    outputs: list[str] = []
    for index in range(max_samples):
        prompt = prompts[index % len(prompts)]
        params = SamplingParams(max_tokens=max_tokens, temperature=0.8, top_p=0.95, seed=rng.randint(0, 1_000_000))
        response = sampling_client.sample(
            ModelInput.from_ints(prompt.prompt_tokens or []), num_samples=1, sampling_params=params
        ).result()
        if response.sequences:
            text = str(response.sequences[0].text or "").strip()
            if text:
                outputs.append(text)
    return outputs


def _anchor_datums(training: Any, seed: int, *, size: int, positives: list[str] | None = None) -> list[Datum]:
    """Anchor the policy to the user's own text (plain continuation datums).

    Anchoring to synthetic strings would pull the policy toward junk; the
    anchor's entire purpose is to keep RL close to the voice SFT learned.
    """
    tokenizer = training.get_tokenizer().result()
    rng = random.Random(seed)
    texts = [t for t in (positives or []) if t.strip()]
    if not texts:
        return []
    sample = texts if len(texts) <= size else rng.sample(texts, size)
    datums: list[Datum] = []
    for passage_text in sample:
        tokens = _encode_prompt(tokenizer, passage_text)
        if len(tokens) < 2:
            continue
        datums.append(Datum(model_input=ModelInput.from_ints(tokens), loss_fn_inputs={"labels": tokens}))
    return datums


def _managed_training_client(
    service: Any, *, name: str | None = None, checkpoint: str | None = None, metadata: dict[str, Any] | None = None
):
    if checkpoint is None and hasattr(service, "training_run"):
        return service.training_run(name=name, metadata=metadata)
    if checkpoint is not None and hasattr(service, "create_training_client_from_state"):

        @contextmanager
        def _context():
            client = service.create_training_client_from_state(checkpoint, metadata=metadata)
            try:
                yield client
            finally:
                _close_client(client)

        return _context()

    @contextmanager
    def _fallback():
        client = service.create_lora_training_client(metadata=metadata)
        try:
            yield client
        finally:
            _close_client(client)

    return _fallback()


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _runtime_model_base(service: Any) -> str | None:
    config = getattr(service, "config", None)
    if config is None:
        return None
    model = getattr(config, "model", None)
    return getattr(model, "base_model", None)


def _encode_prompt(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=True)
    if isinstance(encoded, dict):
        tokens = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        tokens = encoded.input_ids
    else:
        tokens = encoded
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(token) for token in tokens]


def _fluency_score(sampling_client: Any, text: str) -> float:
    tokenizer = sampling_client.get_tokenizer().result()
    tokens = _encode_prompt(tokenizer, text)
    if len(tokens) < 2:
        return 0.0
    logprobs = sampling_client.compute_logprobs(ModelInput.from_ints(tokens)).result()
    values = [float(value) for value in logprobs[1:] if value is not None]
    return sum(values) / len(values) if values else 0.0


def _parse_requested_words(prompt: str) -> int:
    match = re.search(r"about\s+(\d+)\s+words?", prompt, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    words = len(prompt.split())
    return 40 if words < 20 else 100


def _split_passages(passages: list[Passage], *, seed: int) -> tuple[list[Passage], list[Passage]]:
    if not passages:
        return [], []
    rng = random.Random(seed)
    items = list(passages)
    rng.shuffle(items)
    holdout = max(1, len(items) // 10)
    return items[:holdout], items[holdout:]


class _ByteTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [byte for byte in text.encode("utf-8", errors="replace")]}


def _byte_tokenizer() -> _ByteTokenizer:
    return _ByteTokenizer()


def _load_or_eval_store_root(service: Any) -> str:
    config = getattr(service, "config", None)
    if config is None:
        return "."
    return getattr(config, "store_root", ".")


def _store_root(service: Any) -> str:
    return _load_or_eval_store_root(service)


def _latest_run_for_stage(store: RunStore, stage: str):
    for run in store.list_runs(app="scribe"):
        if not isinstance(run.metadata, dict):
            continue
        if run.metadata.get("stage") != stage or run.status != "completed":
            continue
        return run
    return None


def _latest_checkpoint_for_stage(store: RunStore, stage: str) -> str | None:
    run = _latest_run_for_stage(store, stage)
    return _latest_checkpoint_for_run(store, run.id) if run is not None else None


def _latest_checkpoint_for_run(store: RunStore, run_id: str) -> str | None:
    checkpoints = store.list_checkpoints(run_id=run_id)
    return checkpoints[-1].path if checkpoints else None


def _max_round(store: RunStore) -> int:
    rounds = []
    for run in store.list_runs(app="scribe"):
        if not isinstance(run.metadata, dict):
            continue
        if run.metadata.get("stage") != "rl" or run.status != "completed":
            continue
        round_value = run.metadata.get("round")
        if isinstance(round_value, int):
            rounds.append(round_value)
    return max(rounds, default=0)


def _run_summary(store: RunStore, run: Any) -> dict[str, Any]:
    if run is None:
        return {}
    return {
        "run_id": run.id,
        "metadata": dict(run.metadata),
        "checkpoint": _latest_checkpoint_for_run(store, run.id),
    }


def _should_run(steps: list[str] | None, name: str) -> bool:
    return steps is None or name in steps


def _mean(values) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0
