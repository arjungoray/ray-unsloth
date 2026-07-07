"""CLI mounting for the Scribe app."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

from ray_unsloth import ModelInput, SamplingParams
from ray_unsloth.config import load_config
from ray_unsloth.store import RunStore
from ray_unsloth_apps.scribe.ingest import Passage, ingest_paths, load_passages, save_passages, scrub
from ray_unsloth_apps.scribe.pipeline import ScribeConfig, run_pipeline, stage_eval, stage_export
from ray_unsloth_apps.scribe.profile import build_profile, capability_report, load_profile, save_profile
from ray_unsloth_apps.scribe.prompts import task_bank


def build_cli(subparsers: argparse._SubParsersAction[Any]) -> None:
    ingest = subparsers.add_parser("ingest", help="Build the local Scribe corpus.")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--workdir", default="./scribe/")
    ingest.add_argument("--scrub", action="store_true")
    ingest.set_defaults(func=cmd_ingest)

    profile = subparsers.add_parser("profile", help="Build the style profile from the ingested corpus.")
    profile.add_argument("--workdir", default="./scribe/")
    profile.set_defaults(func=cmd_profile)

    train = subparsers.add_parser("train", help="Run SFT + RL using the current workdir state.")
    train.add_argument("--workdir", default="./scribe/")
    train.add_argument("--skip-sft", action="store_true")
    train.add_argument("--rounds", type=int)
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("eval", help="Evaluate the latest checkpoint.")
    evaluate.add_argument("--workdir", default="./scribe/")
    evaluate.set_defaults(func=cmd_eval)

    export = subparsers.add_parser("export", help="Export the latest checkpoint.")
    export.add_argument("--workdir", default="./scribe/")
    export.add_argument("--target", default="ollama")
    export.set_defaults(func=cmd_export)

    pipeline = subparsers.add_parser("pipeline", help="Run the full Scribe pipeline.")
    pipeline.add_argument("paths", nargs="+")
    pipeline.add_argument("--workdir", default="./scribe/")
    pipeline.set_defaults(func=cmd_pipeline)

    mirror = subparsers.add_parser(
        "mirror", help="Terminal game that compares a held-out passage against a generation."
    )
    mirror.add_argument("--workdir", default="./scribe/")
    mirror.add_argument("--pairs", type=int, default=6)
    mirror.set_defaults(func=cmd_mirror)


def cmd_ingest(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).expanduser()
    corpus_path = workdir / "corpus" / "passages.jsonl"
    passages = ingest_paths(args.paths)
    if args.scrub:
        passages = [Passage(text=scrub(passage.text), source=passage.source, kind=passage.kind) for passage in passages]
    save_passages(passages, corpus_path)
    print(json.dumps({"corpus_path": str(corpus_path), "n_passages": len(passages)}, indent=2, sort_keys=True))
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).expanduser()
    passages = _load_passages(workdir)
    profile = build_profile(passages)
    save_profile(profile, workdir / "profile.json")
    print(capability_report(profile))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    runtime = load_config(args.config) if getattr(args, "config", None) else load_config(None)
    config = ScribeConfig.from_yaml(args.config)
    if args.rounds is not None:
        config = replace(config, rl_rounds=args.rounds)
    runtime.scribe = {**runtime.scribe, "rl_rounds": config.rl_rounds}
    steps = ["rl"] if args.skip_sft else ["sft", "rl"]
    run_pipeline(runtime, args.workdir, steps=steps)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    runtime = load_config(args.config) if getattr(args, "config", None) else load_config(None)
    config = ScribeConfig.from_yaml(args.config)
    workdir = Path(args.workdir).expanduser()
    passages = _load_passages(workdir)
    profile = _load_profile(workdir)
    checkpoint = _latest_checkpoint(runtime)
    if checkpoint is None:
        raise ValueError("No completed Scribe checkpoint found.")
    from ray_unsloth import ServiceClient

    service = ServiceClient(config=runtime)
    try:
        report = stage_eval(service, checkpoint, passages, profile, config)
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        service.close()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    runtime = load_config(args.config) if getattr(args, "config", None) else load_config(None)
    checkpoint = _latest_checkpoint(runtime)
    if checkpoint is None:
        raise ValueError("No completed Scribe checkpoint found.")
    report = stage_export(checkpoint, target=args.target, runtime=runtime, workdir=args.workdir)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    run_pipeline(args.config, args.workdir, steps=None, paths=list(args.paths))
    return 0


def cmd_mirror(args: argparse.Namespace) -> int:
    runtime = load_config(args.config) if getattr(args, "config", None) else load_config(None)
    config = ScribeConfig.from_yaml(args.config)
    workdir = Path(args.workdir).expanduser()
    passages = _load_passages(workdir)
    _load_profile(workdir)
    checkpoint = _latest_checkpoint(runtime)
    if checkpoint is None:
        raise ValueError("No completed Scribe checkpoint found.")
    from ray_unsloth import ServiceClient

    service = ServiceClient(config=runtime)
    sampler = service.create_sampling_client(model_path=checkpoint)
    bank = task_bank()
    rng = random.Random(config.seed)
    try:
        fool_count = 0
        for index in range(max(1, args.pairs)):
            passage = passages[index % len(passages)]
            generated = _generation_for_mirror(
                sampler,
                bank[index % len(bank)],
                config.max_tokens,
                seed=rng.randint(0, 1_000_000),
            )
            left, right = _shuffle_pair(passage.text, generated, rng)
            print(f"Pair {index + 1}:")
            print(f"A: {left}")
            print(f"B: {right}")
            answer = input("Which is the held-out passage? [A/B] ").strip().upper()
            actual = "A" if left == passage.text else "B"
            if answer != actual:
                fool_count += 1
        print(f"fool-rate: {fool_count}/{max(1, args.pairs)}")
    finally:
        service.close()
    return 0


def _load_passages(workdir: Path) -> list[Passage]:
    path = workdir / "corpus" / "passages.jsonl"
    if not path.exists():
        raise ValueError(f"Missing corpus file: {path}")
    return load_passages(path)


def _load_profile(workdir: Path):
    path = workdir / "profile.json"
    if not path.exists():
        passages = _load_passages(workdir)
        profile = build_profile(passages)
        save_profile(profile, path)
        return profile
    return load_profile(path)


def _latest_checkpoint(runtime) -> str | None:
    store = RunStore(runtime.store_root)
    for stage in ("rl", "sft"):
        for run in store.list_runs(app="scribe"):
            if not isinstance(run.metadata, dict):
                continue
            if run.metadata.get("stage") != stage or run.status != "completed":
                continue
            checkpoints = store.list_checkpoints(run_id=run.id)
            if checkpoints:
                return checkpoints[-1].path
    return None


def _generation_for_mirror(sampler: Any, prompt_text: str, max_tokens: int, *, seed: int) -> str:
    tokenizer = sampler.get_tokenizer().result()
    encoded = tokenizer(prompt_text, add_special_tokens=True)
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
    response = sampler.sample(
        ModelInput.from_ints([int(token) for token in tokens]),
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.8, top_p=0.95, seed=seed),
    ).result()
    return str(response.sequences[0].text or "") if response.sequences else ""


def _shuffle_pair(left: str, right: str, rng: random.Random) -> tuple[str, str]:
    if rng.random() < 0.5:
        return left, right
    return right, left
