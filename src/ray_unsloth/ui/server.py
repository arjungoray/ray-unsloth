"""FastAPI UI/API over local ray-unsloth run data."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ray_unsloth.config import RuntimeConfig, load_config
from ray_unsloth.plugins import apps as app_registry
from ray_unsloth.providers import GPU_CATALOG, get_provider, list_providers, resolve_provider_name
from ray_unsloth.store import RunStore


def create_app(config: RuntimeConfig | str | dict[str, Any] | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install the UI extra first: pip install -e '.[ui]'") from exc

    loaded = load_config(config)
    store = RunStore(loaded.store_root)
    app = FastAPI(title="ray-unsloth control plane", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return Path(__file__).with_name("static").joinpath("index.html").read_text()

    @app.get("/api/summary")
    def summary():
        runs = store.list_runs()
        checkpoints = store.list_checkpoints()
        evals = store.list_evals()
        return {
            "provider": resolve_provider_name(loaded),
            "checkpoint_root": loaded.checkpoint_root,
            "runs": len(runs),
            "running": sum(1 for run in runs if run.status == "running"),
            "checkpoints": len(checkpoints),
            "evals": len(evals),
        }

    @app.get("/api/runs")
    def runs():
        return [run.to_dict() for run in store.list_runs()]

    @app.get("/api/apps")
    def apps():
        rows = []
        for name, manifest in app_registry.items():
            runs = [run.to_dict() for run in store.list_runs(app=name)]
            rows.append(
                {
                    "name": manifest.name,
                    "description": manifest.description,
                    "stages": [stage.to_dict() for stage in manifest.stages],
                    "requires": list(manifest.requires),
                    "runs": runs,
                }
            )
        return rows

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str, after: float | None = None):
        record = store.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {
            "run": record.to_dict(),
            "metrics": store.read_metrics(run_id, after=after),
            "logs": store.read_logs(run_id, after=after),
            "checkpoints": [item.to_dict() for item in store.list_checkpoints(run_id=run_id)],
        }

    @app.get("/api/checkpoints")
    def checkpoints():
        return [record.to_dict() for record in store.list_checkpoints()]

    @app.get("/api/checkpoints/lineage")
    def checkpoint_lineage(path: str):
        return [record.to_dict() for record in store.lineage(path)]

    @app.get("/api/evals")
    def evals():
        return store.list_evals()

    @app.get("/api/providers")
    def providers():
        rows = []
        for name in list_providers():
            provider = get_provider(name)
            health = provider.health(loaded)
            rows.append(
                {
                    "name": name,
                    "capabilities": provider.capabilities().to_dict(),
                    "health": asdict(health),
                    "selected": name == resolve_provider_name(loaded),
                }
            )
        return rows

    @app.get("/api/config")
    def config_snapshot():
        return {
            "provider": resolve_provider_name(loaded),
            "checkpoint_root": loaded.checkpoint_root,
            "model": asdict(loaded.model),
            "lora": asdict(loaded.lora),
            "resources": asdict(loaded.resources),
        }

    @app.get("/api/config/validate")
    def validate():
        return [
            {
                "severity": issue.severity,
                "path": issue.path,
                "message": issue.message,
                "hint": issue.hint,
            }
            for issue in loaded.validate()
        ]

    @app.get("/api/losses")
    def losses():
        from ray_unsloth.losses import list_losses

        return [spec.to_dict() for spec in list_losses()]

    @app.get("/api/exporters")
    def exporters():
        from ray_unsloth.plugins import exporters as exporter_registry

        return [{"name": name, "description": exporter_registry.describe(name)} for name in exporter_registry.names()]

    @app.get("/api/topology")
    def topology():
        provider = get_provider(resolve_provider_name(loaded))
        plan = provider.plan(loaded)
        return {
            "provider": provider.name,
            "steps": plan.steps,
            "fit": asdict(plan.fit) if plan.fit is not None else None,
            "gpu_catalog": GPU_CATALOG,
        }

    return app


def serve(
    config: RuntimeConfig | str | dict[str, Any] | None = None, *, host: str = "127.0.0.1", port: int = 8765
) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install the UI extra first: pip install -e '.[ui]'") from exc

    uvicorn.run(create_app(config), host=host, port=port)
