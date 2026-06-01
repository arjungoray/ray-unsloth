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
    elif raw.startswith("tinker://local/"):
        raw = raw.removeprefix("tinker://local/")
        if not raw.startswith("/"):
            raw = f"/{raw}"
    elif raw.startswith("tinker://"):
        raw = str(Path("checkpoints") / "tinker" / raw.removeprefix("tinker://"))
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
    checkpoint_dir = resolve_path(path)
    if not checkpoint_dir.exists():
        raise CheckpointError(f"Checkpoint path does not exist: {checkpoint_dir}")
    manifest_path = checkpoint_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise CheckpointError(f"Missing checkpoint manifest: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"Malformed checkpoint manifest at {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CheckpointError(
            f"Malformed checkpoint manifest at {manifest_path}: expected a JSON object, "
            f"got {type(manifest).__name__}."
        )
    return manifest


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


def validate_restore_manifest(
    manifest: dict[str, Any],
    *,
    path: str | Path,
    base_model: str,
    lora_rank: int,
    target_modules: list[str] | None = None,
) -> None:
    """Validate a checkpoint manifest against the active config before loading weights.

    Raises :class:`CheckpointError` if the manifest is malformed or any key field
    (``base_model``, ``lora.rank``, ``lora.target_modules``) disagrees with the active
    configuration. Fields absent from the manifest are skipped so older checkpoints
    remain loadable.
    """

    if not isinstance(manifest, dict):
        raise CheckpointError(
            f"Malformed checkpoint manifest for '{path}': expected a JSON object, "
            f"got {type(manifest).__name__}."
        )

    manifest_base_model = manifest.get("base_model")
    if manifest_base_model is not None and manifest_base_model != base_model:
        raise CheckpointError(
            f"base_model mismatch for checkpoint '{path}': active config base_model is "
            f"'{base_model}', but the checkpoint manifest was saved from base_model "
            f"'{manifest_base_model}'. Restore with base_model matching the checkpoint manifest "
            "or load a checkpoint saved from the active base_model."
        )

    manifest_lora = manifest.get("lora")
    if manifest_lora is None:
        return
    if not isinstance(manifest_lora, dict):
        raise CheckpointError(
            f"Malformed checkpoint manifest for '{path}': field 'lora' must be a JSON object, "
            f"got {type(manifest_lora).__name__}."
        )

    manifest_rank = manifest_lora.get("rank")
    if manifest_rank is not None and manifest_rank != lora_rank:
        raise CheckpointError(
            f"lora.rank mismatch for checkpoint '{path}': active config lora.rank is {lora_rank}, "
            f"but the checkpoint manifest was saved with rank {manifest_rank}. Set lora.rank to the "
            "checkpoint rank or load weights saved with the configured rank."
        )

    if target_modules is not None:
        manifest_target_modules = manifest_lora.get("target_modules")
        if manifest_target_modules is not None and list(manifest_target_modules) != list(target_modules):
            raise CheckpointError(
                f"lora.target_modules mismatch for checkpoint '{path}': active config "
                f"lora.target_modules is {list(target_modules)!r}, but the checkpoint manifest was "
                f"saved with {list(manifest_target_modules)!r}. Set lora.target_modules to match the "
                "checkpoint or load weights saved for the configured target modules."
            )
