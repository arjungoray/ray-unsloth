"""Checkpoint helpers with atomic manifests."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from ray_unsloth.errors import CheckpointError
from ray_unsloth.types import CheckpointRef

MANIFEST_NAME = "manifest.json"


def resolve_path(uri: str | Path) -> Path:
    raw = str(uri)
    if raw.startswith("local://"):
        raw = raw.removeprefix("local://")
    return Path(raw).expanduser().resolve()


def new_checkpoint_path(root: str | Path, prefix: str, step: int) -> Path:
    return resolve_path(root) / f"{prefix}-step-{step}-{uuid.uuid4().hex[:8]}"


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    checkpoint_path = resolve_path(path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_path / MANIFEST_NAME
    tmp_path = manifest_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, manifest_path)


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = resolve_path(path) / MANIFEST_NAME
    if not manifest_path.exists():
        raise CheckpointError(f"Missing checkpoint manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_checkpoint_dir(target: str | Path):
    """Context manager that publishes a directory via atomic rename."""

    class _AtomicCheckpointDir:
        def __init__(self, final_path: Path):
            self.final_path = final_path
            self.tmp_path: Path | None = None

        def __enter__(self) -> Path:
            self.final_path.parent.mkdir(parents=True, exist_ok=True)
            self.tmp_path = Path(
                tempfile.mkdtemp(
                    prefix=f".{self.final_path.name}.",
                    dir=str(self.final_path.parent),
                )
            )
            return self.tmp_path

        def __exit__(self, exc_type, exc, tb) -> bool:
            if self.tmp_path is None:
                return False
            if exc_type is not None:
                shutil.rmtree(self.tmp_path, ignore_errors=True)
                return False
            if self.final_path.exists():
                shutil.rmtree(self.final_path)
            os.replace(self.tmp_path, self.final_path)
            return False

    return _AtomicCheckpointDir(resolve_path(target))


def checkpoint_ref(path: str | Path, has_optimizer: bool) -> CheckpointRef:
    manifest = read_manifest(path)
    return CheckpointRef(
        path=str(resolve_path(path)),
        step=manifest.get("step"),
        has_optimizer=has_optimizer,
        metadata=manifest,
    )


def base_manifest(
    *,
    kind: str,
    step: int,
    base_model: str,
    lora: dict[str, Any],
    has_optimizer: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "kind": kind,
        "step": step,
        "base_model": base_model,
        "lora": lora,
        "has_optimizer": has_optimizer,
        "created_at": time.time(),
    }
    if extra:
        manifest.update(extra)
    return manifest
