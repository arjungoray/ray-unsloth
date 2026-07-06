"""Command-line workflows for ray-unsloth."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient
from ray_unsloth.config import RuntimeConfig, load_config
from ray_unsloth.evals import EvalSpec, run_eval
from ray_unsloth.export import export_checkpoint, list_exporters
from ray_unsloth.providers import get_provider, list_providers, resolve_provider_name
from ray_unsloth.store import RunStore


def _issue_dict(issue: Any) -> dict[str, Any]:
    return {
        "severity": getattr(issue, "severity", "info"),
        "path": getattr(issue, "path", ""),
        "message": getattr(issue, "message", str(issue)),
        "hint": getattr(issue, "hint", None),
    }


def _load(args: argparse.Namespace) -> RuntimeConfig:
    return load_config(args.config) if getattr(args, "config", None) else RuntimeConfig()


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to overwrite.", file=sys.stderr)
        return 2
    config = {
        "provider": "fake",
        "run_name": "local-fake-smoke",
        "checkpoint_root": "checkpoints",
        "model": {"base_model": "unsloth/gemma-4-E2B-it", "max_seq_length": 512},
        "lora": {"rank": 8},
        "resources": {"sampler_replicas": 1},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(path)
    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    config = _load(args)
    issues = config.validate()
    if args.json:
        print(json.dumps([_issue_dict(issue) for issue in issues], indent=2, sort_keys=True))
    elif not issues:
        print("Config valid.")
    else:
        for issue in issues:
            print(str(issue))
    return 1 if any(getattr(issue, "severity", "") == "error" for issue in issues) else 0


def cmd_list_providers(args: argparse.Namespace) -> int:
    config = _load(args)
    rows = []
    for name in list_providers():
        provider = get_provider(name)
        health = provider.health(config)
        cap = provider.capabilities()
        rows.append({"name": name, "kind": cap.kind, "health": health.detail, "description": cap.description})
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(f"{row['name']:10} {row['kind']:9} {row['health']:24} {row['description']}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    config = _load(args)
    provider_name = args.provider or resolve_provider_name(config)
    if args.provider:
        config.provider = args.provider
    plan = get_provider(provider_name).plan(config)
    if args.json:
        payload = {
            "provider": plan.provider,
            "summary": plan.summary,
            "steps": plan.steps,
            "artifacts": plan.artifacts,
            "estimated_hourly_cost_usd": plan.estimated_hourly_cost_usd,
            "fit": asdict(plan.fit) if plan.fit is not None else None,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(plan.render())
        if args.write_artifacts and plan.artifacts:
            out_dir = Path(args.write_artifacts)
            out_dir.mkdir(parents=True, exist_ok=True)
            for filename, content in plan.artifacts.items():
                (out_dir / filename).write_text(content)
            print(f"\nWrote artifacts to {out_dir}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = _load(args)
    provider = get_provider(resolve_provider_name(config))
    issues = config.validate()
    health = provider.health(config)
    print(f"provider: {provider.name}")
    print(f"health: {health.detail}")
    if issues:
        for issue in issues:
            print(str(issue))
    else:
        print("config: valid")
    return 1 if not health.ok or any(getattr(issue, "severity", "") == "error" for issue in issues) else 0


def _datum_from_text(text: str) -> Datum:
    tokens = list(text.encode("utf-8", errors="replace"))
    return Datum(model_input=ModelInput.from_ints(tokens), loss_fn_inputs={"labels": tokens})


def cmd_run(args: argparse.Namespace) -> int:
    config = _load(args)
    service = ServiceClient(config=config)
    trainer = service.create_lora_training_client(seed=args.seed)
    for _ in range(args.steps):
        trainer.forward_backward([_datum_from_text(args.text)]).result()
        trainer.optim_step(AdamParams(learning_rate=args.learning_rate)).result()
    checkpoint = trainer.save_state(args.checkpoint_name).result()
    print(json.dumps({"run_id": trainer.run_id, "checkpoint": checkpoint.path, "step": checkpoint.step}, indent=2))
    service.close()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    config = _load(args)
    service = ServiceClient(config=config)
    sampler = service.create_sampling_client(model_path=args.checkpoint)
    store = RunStore(config.store_root)
    spec = EvalSpec(
        name=args.name,
        dataset=args.dataset,
        scorer=args.scorer,
        max_samples=args.max_samples,
        checkpoint_path=args.checkpoint,
        sampling_params=SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature),
    )
    report = run_eval(sampler, spec, store=store)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    service.close()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    options: dict[str, Any] = {}
    for raw in getattr(args, "option", []) or []:
        key, sep, value = raw.partition("=")
        if not sep:
            print(f"Invalid --option '{raw}': expected KEY=VALUE.", file=sys.stderr)
            return 2
        options[key] = {"true": True, "false": False}.get(value.lower(), value)
    report = export_checkpoint(args.target, args.checkpoint, output=args.output, **options)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    config = _load(args)
    store = RunStore(config.store_root)
    runs = [run.to_dict() for run in store.list_runs()]
    if args.json:
        print(json.dumps(runs, indent=2, sort_keys=True))
    else:
        for run in runs:
            print(f"{run['id']} {run['status']} {run['provider']} {run['base_model']}")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    root = Path(load_config(args.config).checkpoint_root if args.config else args.path).expanduser()
    if args.yes:
        import shutil

        shutil.rmtree(root / "_store", ignore_errors=True)
        print(f"Removed {root / '_store'}")
        return 0
    print(f"Would remove {root / '_store'}; pass --yes to confirm.")
    return 0


def cmd_serve_ui(args: argparse.Namespace) -> int:
    from ray_unsloth.ui.server import serve

    config = _load(args)
    serve(config, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ray-unsloth")
    parser.add_argument("--config", help="Path to a YAML runtime config.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("path", nargs="?", default="ray-unsloth.yaml")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    validate = sub.add_parser("validate-config")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=cmd_validate_config)

    providers = sub.add_parser("list-providers")
    providers.add_argument("--json", action="store_true")
    providers.set_defaults(func=cmd_list_providers)

    plan = sub.add_parser("plan")
    plan.add_argument("--provider")
    plan.add_argument("--json", action="store_true")
    plan.add_argument("--write-artifacts")
    plan.set_defaults(func=cmd_plan)

    doctor = sub.add_parser("doctor")
    doctor.set_defaults(func=cmd_doctor)

    run = sub.add_parser("run")
    run.add_argument("--steps", type=int, default=3)
    run.add_argument("--text", default="ray-unsloth trains open post-training workflows.")
    run.add_argument("--learning-rate", type=float, default=0.2)
    run.add_argument("--seed", type=int, default=3407)
    run.add_argument("--checkpoint-name", default=None)
    run.set_defaults(func=cmd_run)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("dataset")
    evaluate.add_argument("--name", default="eval")
    evaluate.add_argument("--scorer", default="contains")
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--max-tokens", type=int, default=64)
    evaluate.add_argument("--temperature", type=float, default=0.0)
    evaluate.set_defaults(func=cmd_eval)

    export = sub.add_parser("export")
    export.add_argument("checkpoint")
    export.add_argument("--target", default="local", choices=list_exporters())
    export.add_argument("--output")
    export.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Exporter-specific option (repeatable), e.g. --option llama_cpp_dir=~/llama.cpp --option create=true",
    )
    export.set_defaults(func=cmd_export)

    runs = sub.add_parser("runs")
    runs.add_argument("--json", action="store_true")
    runs.set_defaults(func=cmd_runs)

    clean = sub.add_parser("clean")
    clean.add_argument("--path", default="checkpoints")
    clean.add_argument("--yes", action="store_true")
    clean.set_defaults(func=cmd_clean)

    serve_ui = sub.add_parser("serve-ui")
    serve_ui.add_argument("--host", default="127.0.0.1")
    serve_ui.add_argument("--port", type=int, default=8765)
    serve_ui.set_defaults(func=cmd_serve_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
