"""Command-line workflows for ray-unsloth."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import tomllib
import yaml

from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient
from ray_unsloth.apps import app_install_status, list_apps
from ray_unsloth.config import RuntimeConfig, load_config
from ray_unsloth.errors import ERROR_CATALOG
from ray_unsloth.evals import EvalSpec, run_eval
from ray_unsloth.export import export_checkpoint, list_exporters
from ray_unsloth.plugins import load_entry_point_plugins
from ray_unsloth.providers import get_provider, list_providers, resolve_provider_name
from ray_unsloth.schema import config_json_schema
from ray_unsloth.store import RunStore

SPARKLINE_BARS = "▁▂▃▄▅▆▇█"
_VERSION_PARTS = re.compile(r"\d+")


def _issue_dict(issue: Any) -> dict[str, Any]:
    return {
        "severity": getattr(issue, "severity", "info"),
        "path": getattr(issue, "path", ""),
        "message": getattr(issue, "message", str(issue)),
        "hint": getattr(issue, "hint", None),
    }


def _load(args: argparse.Namespace) -> RuntimeConfig:
    return load_config(args.config) if getattr(args, "config", None) else RuntimeConfig()


def _write_init_config(path: Path) -> None:
    config = {
        "provider": "fake",
        "run_name": "local-fake-smoke",
        "checkpoint_root": "checkpoints",
        "model": {"base_model": "unsloth/gemma-4-E2B-it", "max_seq_length": 512},
        "lora": {"rank": 8},
        "resources": {"sampler_replicas": 1},
    }
    schema_path = path.with_name("ray-unsloth.schema.json")
    path.write_text(
        "# yaml-language-server: $schema=./ray-unsloth.schema.json\n" + yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    schema_path.write_text(json.dumps(config_json_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to overwrite.", file=sys.stderr)
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_init_config(path)
    print(path)
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    del args
    print(json.dumps(config_json_schema(), indent=2, sort_keys=True))
    return 0


def cmd_errors(args: argparse.Namespace) -> int:
    rows = [{"code": code, "description": description} for code, description in sorted(ERROR_CATALOG.items())]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(f"{'CODE':<8} DESCRIPTION")
        for row in rows:
            print(f"{row['code']:<8} {row['description']}")
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
                (out_dir / filename).write_text(content, encoding="utf-8")
            print(f"\nWrote artifacts to {out_dir}")
    return 0


def _python_version_row() -> dict[str, Any]:
    version = sys.version.split()[0]
    ok = sys.version_info >= (3, 10)
    return {
        "name": "python",
        "status": "PASS" if ok else "FAIL",
        "detail": f"{version} (requires >=3.10)",
        "fix": None if ok else "python3.10 -m venv .venv && . .venv/bin/activate",
    }


def _torch_row() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "name": "torch",
            "status": "FAIL",
            "detail": "torch is not importable",
            "fix": "pip install -e '.[dev]'",
        }
    cuda = bool(getattr(torch.cuda, "is_available", lambda: False)())
    mps = bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())
    detail = f"torch {getattr(torch, '__version__', 'unknown')}; CUDA={cuda}; MPS={mps}"
    if cuda or mps:
        status = "PASS"
        fix = None
    else:
        status = "WARN"
        fix = "Use a GPU-enabled host or switch to provider: fake"
    return {"name": "torch+accelerator", "status": status, "detail": detail, "fix": fix}


def _gpu_names_row() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"name": "gpu names", "status": "WARN", "detail": "torch is unavailable", "fix": "pip install torch"}
    if not getattr(torch.cuda, "is_available", lambda: False)():
        return {
            "name": "gpu names",
            "status": "WARN",
            "detail": "no CUDA devices visible",
            "fix": "Use a CUDA host or provider: fake",
        }
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {"name": "gpu names", "status": "PASS", "detail": ", ".join(names), "fix": None}


def _modal_profile_current() -> str | None:
    try:
        result = subprocess.run(
            ["modal", "profile", "current"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    profile = result.stdout.strip()
    return profile or None


def _modal_row() -> dict[str, Any]:
    package_ok = importlib.util.find_spec("modal") is not None
    profile = _modal_profile_current()
    if package_ok and profile:
        return {
            "name": "modal",
            "status": "PASS",
            "detail": f"package installed; current profile: {profile}",
            "fix": None,
        }
    missing = []
    if not package_ok:
        missing.append("package")
    if not profile:
        missing.append("profile")
    return {
        "name": "modal",
        "status": "WARN",
        "detail": f"missing {', '.join(missing)}",
        "fix": "pip install -e '.[modal]' && modal setup",
    }


def _hf_token_row() -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token()
        except Exception:
            token = None
    if token:
        return {"name": "hf token", "status": "PASS", "detail": "token detected", "fix": None}
    return {
        "name": "hf token",
        "status": "WARN",
        "detail": "no Hugging Face token found",
        "fix": "huggingface-cli login",
    }


def _disk_row(config: RuntimeConfig) -> dict[str, Any]:
    checkpoint_root = Path(config.checkpoint_root).expanduser()
    usage_path = checkpoint_root if checkpoint_root.exists() else checkpoint_root.parent
    usage = shutil.disk_usage(usage_path)
    free_gb = usage.free / (1024**3)
    if free_gb >= 5:
        status = "PASS"
        fix = None
    else:
        status = "WARN"
        fix = f"rm -rf {Path(config.checkpoint_root).expanduser() / '_store'}"
    return {
        "name": "disk space",
        "status": status,
        "detail": f"{free_gb:.1f} GB free under checkpoint_root",
        "fix": fix,
    }


def _config_row(config: RuntimeConfig, args: argparse.Namespace) -> dict[str, Any]:
    issues = config.validate()
    errors = [issue for issue in issues if getattr(issue, "severity", "") == "error"]
    warnings = [issue for issue in issues if getattr(issue, "severity", "") == "warning"]
    if errors:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"
    detail = (
        "valid"
        if not issues
        else "; ".join(
            f"{getattr(issue, 'severity', 'info')}:{getattr(issue, 'path', '')} {getattr(issue, 'message', issue)}"
            for issue in issues
        )
    )
    fix = "ray-unsloth validate-config"
    if getattr(args, "config", None):
        fix += f" --config {args.config}"
    return {"name": "config", "status": status, "detail": detail, "fix": None if status == "PASS" else fix}


def _provider_health_row(config: RuntimeConfig) -> dict[str, Any]:
    try:
        provider = get_provider(resolve_provider_name(config))
        health = provider.health(config)
    except Exception as exc:
        return {
            "name": "provider health",
            "status": "FAIL",
            "detail": str(exc),
            "fix": "ray-unsloth plan --provider fake",
        }
    return {
        "name": "provider health",
        "status": "PASS" if health.ok else "FAIL",
        "detail": health.detail,
        "fix": None if health.ok else f"ray-unsloth plan --provider {provider.name}",
    }


def _dependency_parts(version: str) -> tuple[int, ...]:
    parts = [int(piece) for piece in _VERSION_PARTS.findall(version)]
    return tuple(parts[:4]) if parts else (0,)


def _satisfies_version(version: str, spec: str) -> bool | None:
    if not spec:
        return True
    current = _dependency_parts(version)
    for clause in spec.split(","):
        clause = clause.strip()
        if not clause:
            continue
        if clause.startswith(">="):
            if current < _dependency_parts(clause[2:]):
                return False
        elif clause.startswith("<="):
            if current > _dependency_parts(clause[2:]):
                return False
        elif clause.startswith(">"):
            if current <= _dependency_parts(clause[1:]):
                return False
        elif clause.startswith("<"):
            if current >= _dependency_parts(clause[1:]):
                return False
        elif clause.startswith("=="):
            if current != _dependency_parts(clause[2:]):
                return False
        elif clause.startswith("!="):
            if current == _dependency_parts(clause[2:]):
                return False
        else:
            return None
    return True


@lru_cache(maxsize=1)
def _pyproject_dependencies() -> dict[str, str]:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = data["project"]["dependencies"]
    specs: dict[str, str] = {}
    for requirement in dependencies:
        name, _, spec = requirement.partition(">=")
        if not spec:
            name, _, spec = requirement.partition("==")
        if not spec:
            name = requirement.split("[", 1)[0]
            spec = ""
        normalized = name.strip().lower().replace("_", "-")
        specs[normalized] = requirement[len(name) :].strip()
    return specs


def _dependency_row() -> dict[str, Any]:
    specs = _pyproject_dependencies()
    rows = []
    status = "PASS"
    for package in ("torch", "transformers"):
        spec = specs.get(package)
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            rows.append(f"{package}: missing")
            status = "WARN"
            continue
        satisfies = _satisfies_version(version, spec or "")
        if satisfies is False:
            rows.append(f"{package} {version} ! {spec or 'unconstrained'}")
            status = "WARN"
        elif satisfies is None:
            rows.append(f"{package} {version} (could not parse {spec or 'spec'})")
            status = "WARN"
        else:
            rows.append(f"{package} {version} satisfies {spec or 'unconstrained'}")
    return {
        "name": "dependency pins",
        "status": status,
        "detail": "; ".join(rows),
        "fix": None if status == "PASS" else "pip install -U torch transformers",
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    config = _load(args)
    rows = [
        _python_version_row(),
        _torch_row(),
        _gpu_names_row(),
        _modal_row(),
        _hf_token_row(),
        _disk_row(config),
        _config_row(config, args),
        _provider_health_row(config),
        _dependency_row(),
    ]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(f"{row['status']:<4} {row['name']:<16} {row['detail']}")
            if row.get("fix"):
                print(f"     fix: {row['fix']}")
    return 1 if any(row["status"] == "FAIL" for row in rows) else 0


def _datum_from_text(text: str) -> Datum:
    tokens = list(text.encode("utf-8", errors="replace"))
    return Datum(model_input=ModelInput.from_ints(tokens), loss_fn_inputs={"labels": tokens})


def _training_text(args: argparse.Namespace) -> str:
    raw = getattr(args, "data", None)
    if raw:
        path = Path(raw).expanduser()
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return raw
    return getattr(args, "text", "ray-unsloth trains open post-training workflows.")


def _sparkline(values: list[float], width: int = 24) -> str:
    if not values:
        return " " * width
    recent = values[-width:]
    low = min(recent)
    high = max(recent)
    if high == low:
        chars = [SPARKLINE_BARS[0]] * len(recent)
    else:
        span = high - low
        chars = [SPARKLINE_BARS[min(len(SPARKLINE_BARS) - 1, int(((value - low) / span) * 7))] for value in recent]
    if len(chars) < width:
        chars = [" "] * (width - len(chars)) + chars
    return "".join(chars)


def _read_losses(store: RunStore | None, run_id: str | None) -> list[float]:
    if store is None or run_id is None:
        return []
    rows = store.read_metrics(run_id)
    losses = []
    for row in rows:
        if row.get("event") != "forward_backward":
            continue
        loss = row.get("loss")
        if isinstance(loss, (int, float)):
            losses.append(float(loss))
    return losses


def _run_training_loop(
    config: RuntimeConfig,
    args: argparse.Namespace,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    service = ServiceClient(config=config)
    try:
        trainer = service.create_lora_training_client(seed=args.seed)
        store = RunStore(config.store_root) if config.tracking else None
        text = _training_text(args)
        datum = _datum_from_text(text)
        losses: list[float] = []
        for step in range(1, args.steps + 1):
            trainer.forward_backward([datum]).result()
            trainer.optim_step(AdamParams(learning_rate=args.learning_rate)).result()
            losses = _read_losses(store, trainer.run_id) or losses
            if progress:
                spark = _sparkline(losses)
                loss_text = f"{losses[-1]:.4f}" if losses else "n/a"
                line = f"step {step}/{args.steps} loss={loss_text} {spark}"
                if sys.stdout.isatty():
                    sys.stdout.write(f"\r\x1b[2K{line}")
                    sys.stdout.flush()
                else:
                    print(line)
        checkpoint = trainer.save_state(args.checkpoint_name).result()
        result = {
            "run_id": trainer.run_id,
            "checkpoint": checkpoint.path,
            "step": checkpoint.step,
        }
        return result
    finally:
        service.close()


def cmd_run(args: argparse.Namespace) -> int:
    config = _load(args)
    result = _run_training_loop(config, args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _modal_profile_available() -> bool:
    return _modal_profile_current() is not None


def _default_up_config(allow_spend: bool) -> RuntimeConfig:
    config = RuntimeConfig()
    try:
        import torch

        cuda_available = bool(getattr(torch.cuda, "is_available", lambda: False)())
    except Exception:
        cuda_available = False
    if cuda_available:
        config.provider = "local-ray"
        return config
    if allow_spend and importlib.util.find_spec("modal") is not None and _modal_profile_available():
        config.provider = "modal"
        config.modal.enabled = True
        return config
    config.provider = "fake"
    return config


def _up_config(args: argparse.Namespace) -> RuntimeConfig:
    config = load_config(args.config) if getattr(args, "config", None) else _default_up_config(args.allow_spend)
    if args.provider:
        config.provider = args.provider
        if args.provider == "modal":
            config.modal.enabled = True
    elif not getattr(args, "config", None) and config.provider == "modal":
        config.modal.enabled = True
    return config


def _print_up_plan(config: RuntimeConfig) -> None:
    provider_name = resolve_provider_name(config)
    plan = get_provider(provider_name).plan(config)
    print(f"Provider: {plan.provider}")
    print(f"Model: {config.model.base_model}")
    print("Steps:")
    for step in plan.steps:
        print(f"- {step}")
    if plan.estimated_hourly_cost_usd is not None:
        print(f"Estimated cost: ~${plan.estimated_hourly_cost_usd:.2f}/hour")


def cmd_up(args: argparse.Namespace) -> int:
    config = _up_config(args)
    _print_up_plan(config)
    if not args.yes:
        response = input("Proceed? [y/N] ").strip().lower()
        if response not in {"y", "yes"}:
            print("Cancelled.")
            return 1
    result = _run_training_loop(config, args, progress=True)
    if sys.stdout.isatty():
        print()
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Run ID: {result['run_id']}")
    serve_ui = "ray-unsloth serve-ui"
    if getattr(args, "config", None):
        serve_ui = f"ray-unsloth --config {args.config} serve-ui"
    print(f"Serve UI: {serve_ui}")
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


def cmd_apps(args: argparse.Namespace) -> int:
    rows = []
    for manifest in list_apps():
        status = app_install_status(manifest)
        rows.append({**manifest.to_dict(), **status})
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            stage_names = ", ".join(stage["name"] for stage in row["stages"])
            suffix = ""
            if row["requires"]:
                suffix = f" requires={','.join(row['requires'])}"
                if row["missing_requires"]:
                    suffix += f" missing={','.join(row['missing_requires'])}"
            print(f"{row['name']:<16} {row['description']} [{stage_names}]{suffix}")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    root = Path(load_config(args.config).checkpoint_root if args.config else args.path).expanduser()
    if args.yes:
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
    load_entry_point_plugins()
    parser = argparse.ArgumentParser(prog="ray-unsloth")
    parser.add_argument("--config", help="Path to a YAML runtime config.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create a starter runtime config and schema file.")
    init.add_argument("path", nargs="?", default="ray-unsloth.yaml")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    errors = sub.add_parser("errors", help="Print the stable error-code catalog.")
    errors.add_argument("--json", action="store_true")
    errors.set_defaults(func=cmd_errors)

    schema = sub.add_parser("schema", help="Print the runtime config JSON Schema.")
    schema.set_defaults(func=cmd_schema)

    validate = sub.add_parser("validate-config", help="Validate the loaded runtime config.")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=cmd_validate_config)

    providers = sub.add_parser("list-providers", help="List registered runtime providers.")
    providers.add_argument("--json", action="store_true")
    providers.set_defaults(func=cmd_list_providers)

    plan = sub.add_parser("plan", help="Render a provider launch plan without running it.")
    plan.add_argument("--provider")
    plan.add_argument("--json", action="store_true")
    plan.add_argument("--write-artifacts")
    plan.set_defaults(func=cmd_plan)

    doctor = sub.add_parser("doctor", help="Run environment, config, and provider readiness checks.")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    run = sub.add_parser("run", help="Run the synchronous fake or real training loop.")
    run.add_argument("--steps", type=int, default=3)
    run.add_argument("--text", default="ray-unsloth trains open post-training workflows.")
    run.add_argument("--data", help="Optional training text or a path to a text file.")
    run.add_argument("--learning-rate", type=float, default=0.2)
    run.add_argument("--seed", type=int, default=3407)
    run.add_argument("--checkpoint-name", default=None)
    run.set_defaults(func=cmd_run)

    up = sub.add_parser("up", help="Auto-detect a provider, confirm the plan, and run the golden path.")
    up.add_argument("--provider")
    up.add_argument("--allow-spend", action="store_true", help="Allow Modal auto-detection to choose a paid runtime.")
    up.add_argument("--yes", action="store_true")
    up.add_argument("--steps", type=int, default=10)
    up.add_argument("--text", default="ray-unsloth trains open post-training workflows.")
    up.add_argument("--data", help="Optional training text or a path to a text file.")
    up.add_argument("--learning-rate", type=float, default=0.2)
    up.add_argument("--seed", type=int, default=3407)
    up.add_argument("--checkpoint-name", default=None)
    up.set_defaults(func=cmd_up)

    evaluate = sub.add_parser("eval", help="Evaluate a saved checkpoint against a dataset.")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("dataset")
    evaluate.add_argument("--name", default="eval")
    evaluate.add_argument("--scorer", default="contains")
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--max-tokens", type=int, default=64)
    evaluate.add_argument("--temperature", type=float, default=0.0)
    evaluate.set_defaults(func=cmd_eval)

    export = sub.add_parser("export", help="Export a checkpoint to a target serving format.")
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

    runs = sub.add_parser("runs", help="List runs from the local store.")
    runs.add_argument("--json", action="store_true")
    runs.set_defaults(func=cmd_runs)

    apps = sub.add_parser("apps", help="List registered ray-unsloth applications.")
    apps.add_argument("--json", action="store_true")
    apps.set_defaults(func=cmd_apps)

    clean = sub.add_parser("clean", help="Remove the local store directory.")
    clean.add_argument("--path", default="checkpoints")
    clean.add_argument("--yes", action="store_true")
    clean.set_defaults(func=cmd_clean)

    serve_ui = sub.add_parser("serve-ui", help="Run the local UI server.")
    serve_ui.add_argument("--host", default="127.0.0.1")
    serve_ui.add_argument("--port", type=int, default=8765)
    serve_ui.set_defaults(func=cmd_serve_ui)

    for manifest in list_apps():
        app_parser = sub.add_parser(manifest.name, help=manifest.description, description=manifest.description)
        app_sub = app_parser.add_subparsers(dest=f"{manifest.name}_command", required=True)
        manifest.build_cli(app_sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
