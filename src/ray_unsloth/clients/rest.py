"""Local filesystem-backed subset of Tinker's REST client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ray_unsloth.checkpoints import MANIFEST_NAME, read_manifest, resolve_path, write_manifest
from ray_unsloth.config import RuntimeConfig
from ray_unsloth.types import (
    Checkpoint,
    CheckpointsListResponse,
    ImmediateFuture,
    TrainingRun,
    TrainingRunsResponse,
    WeightsInfoResponse,
)


class RestClient:
    """Small local checkpoint index for Tinker-style inspection APIs."""

    def __init__(self, *, config: RuntimeConfig):
        self.config = config

    def list_checkpoints(self, training_run_id: str | None = None):
        checkpoints = self._checkpoints(training_run_id=training_run_id)
        return ImmediateFuture(CheckpointsListResponse(checkpoints=checkpoints))

    def list_training_runs(self):
        runs: dict[str, TrainingRun] = {}
        for checkpoint in self._checkpoints():
            run_id = str(checkpoint.metadata.get("session_id") or "local")
            run = runs.setdefault(
                run_id,
                TrainingRun(
                    id=run_id,
                    metadata=dict(checkpoint.metadata.get("metadata") or {}),
                    checkpoints=[],
                ),
            )
            run.checkpoints.append(checkpoint)
        return ImmediateFuture(TrainingRunsResponse(training_runs=list(runs.values())))

    def get_training_run(self, training_run_id: str):
        checkpoints = self._checkpoints(training_run_id=training_run_id)
        metadata = dict(checkpoints[0].metadata.get("metadata") or {}) if checkpoints else {}
        return ImmediateFuture(TrainingRun(id=training_run_id, metadata=metadata, checkpoints=checkpoints))

    def get_weights_info(self, path: str):
        manifest = read_manifest(path)
        return ImmediateFuture(WeightsInfoResponse(path=str(resolve_path(path)), metadata=manifest))

    def publish_checkpoint_from_tinker_path(self, path: str):
        resolved = resolve_path(path)
        manifest = read_manifest(resolved)
        manifest["published"] = True
        write_manifest(resolved, manifest)
        return ImmediateFuture(Checkpoint(path=str(resolved), step=manifest.get("step"), metadata=manifest))

    def _checkpoints(self, training_run_id: str | None = None) -> list[Checkpoint]:
        root = resolve_path(self.config.checkpoint_root)
        if not root.exists():
            return []
        checkpoints = []
        for manifest_path in root.rglob(MANIFEST_NAME):
            checkpoint_path = manifest_path.parent
            try:
                manifest: dict[str, Any] = read_manifest(checkpoint_path)
            except Exception:
                continue
            if training_run_id is not None and manifest.get("session_id") != training_run_id:
                continue
            checkpoints.append(
                Checkpoint(
                    path=str(checkpoint_path),
                    step=manifest.get("step"),
                    metadata=manifest,
                )
            )
        return sorted(checkpoints, key=lambda item: (item.step is None, item.step or 0, item.path))
