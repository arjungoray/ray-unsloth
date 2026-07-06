"""Local run / checkpoint / eval store.

A lightweight, file-backed record of everything that happens on this machine:
training runs, step metrics, checkpoint lineage, and eval reports. Lives
under ``<checkpoint_root>/_store/`` so it travels with the checkpoints it
describes. No database, no daemon — plain JSON + JSONL, safe to read while a
run is writing, and the single source of truth for the CLI (``ray-unsloth
runs``) and the web control plane.

Layout::

    <root>/_store/
      runs/<run_id>.json           run record (status, config snapshot, ...)
      runs/<run_id>.metrics.jsonl  one JSON object per recorded step
      runs/<run_id>.logs.jsonl     structured log lines
      checkpoints.jsonl            global checkpoint index with parent lineage
      evals/<eval_id>.json         eval reports
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ray_unsloth.checkpoints import resolve_path

STORE_DIR = "_store"


@dataclass(slots=True)
class RunRecord:
    id: str
    name: str
    status: str  # "running" | "completed" | "failed"
    provider: str
    base_model: str
    lora_rank: int | None = None
    session_id: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CheckpointRecord:
    path: str
    run_id: str | None
    step: int | None
    kind: str  # "training_state" | "sampler" | ...
    parent: str | None = None
    has_optimizer: bool = False
    created_at: float = 0.0
    base_model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunStore:
    """File-backed store; every method is safe to call from multiple threads."""

    def __init__(self, checkpoint_root: str | Path):
        self.root = resolve_path(checkpoint_root) / STORE_DIR
        self._lock = threading.Lock()

    # -- paths ---------------------------------------------------------------

    def _runs_dir(self) -> Path:
        path = self.root / "runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _evals_dir(self) -> Path:
        path = self.root / "evals"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _checkpoints_file(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / "checkpoints.jsonl"

    def _run_file(self, run_id: str) -> Path:
        return self._runs_dir() / f"{run_id}.json"

    # -- runs -----------------------------------------------------------------

    def create_run(
        self,
        *,
        name: str | None = None,
        provider: str,
        base_model: str,
        lora_rank: int | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> RunRecord:
        run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now = time.time()
        record = RunRecord(
            id=run_id,
            name=name or run_id,
            status="running",
            provider=provider,
            base_model=base_model,
            lora_rank=lora_rank,
            session_id=session_id,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
            config=dict(config or {}),
        )
        self._write_run(record)
        return record

    def _write_run(self, record: RunRecord) -> None:
        path = self._run_file(record.id)
        with self._lock:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True))
            os.replace(tmp, path)

    def get_run(self, run_id: str) -> RunRecord | None:
        path = self._run_file(run_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return RunRecord(**data)

    def list_runs(self) -> list[RunRecord]:
        runs = []
        for path in sorted(self._runs_dir().glob("run-*.json")):
            try:
                runs.append(RunRecord(**json.loads(path.read_text())))
            except (json.JSONDecodeError, TypeError):
                continue
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    def update_run(self, run_id: str, **updates: Any) -> RunRecord | None:
        record = self.get_run(run_id)
        if record is None:
            return None
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)
        record.updated_at = time.time()
        if updates.get("status") in ("completed", "failed") and record.finished_at is None:
            record.finished_at = record.updated_at
        self._write_run(record)
        return record

    # -- metrics / logs ---------------------------------------------------------

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def append_metrics(self, run_id: str, metrics: dict[str, Any]) -> None:
        payload = {"time": time.time(), **metrics}
        self._append_jsonl(self._runs_dir() / f"{run_id}.metrics.jsonl", payload)

    def read_metrics(self, run_id: str, *, after: float | None = None) -> list[dict[str, Any]]:
        path = self._runs_dir() / f"{run_id}.metrics.jsonl"
        if not path.exists():
            return []
        rows = []
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if after is not None and row.get("time", 0) <= after:
                continue
            rows.append(row)
        return rows

    def append_log(self, run_id: str, message: str, *, level: str = "info") -> None:
        self._append_jsonl(
            self._runs_dir() / f"{run_id}.logs.jsonl",
            {"time": time.time(), "level": level, "message": message},
        )

    def read_logs(self, run_id: str, *, after: float | None = None) -> list[dict[str, Any]]:
        path = self._runs_dir() / f"{run_id}.logs.jsonl"
        if not path.exists():
            return []
        rows = []
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if after is not None and row.get("time", 0) <= after:
                continue
            rows.append(row)
        return rows

    # -- checkpoints ------------------------------------------------------------

    def record_checkpoint(
        self,
        *,
        path: str,
        run_id: str | None,
        step: int | None,
        kind: str,
        parent: str | None = None,
        has_optimizer: bool = False,
        base_model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        record = CheckpointRecord(
            path=str(path),
            run_id=run_id,
            step=step,
            kind=kind,
            parent=parent,
            has_optimizer=has_optimizer,
            created_at=time.time(),
            base_model=base_model,
            metadata=dict(metadata or {}),
        )
        self._append_jsonl(self._checkpoints_file(), record.to_dict())
        return record

    def list_checkpoints(self, *, run_id: str | None = None) -> list[CheckpointRecord]:
        path = self._checkpoints_file()
        if not path.exists():
            return []
        records = []
        for line in path.read_text().splitlines():
            try:
                record = CheckpointRecord(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            if run_id is not None and record.run_id != run_id:
                continue
            records.append(record)
        return records

    def lineage(self, checkpoint_path: str) -> list[CheckpointRecord]:
        """The ancestry chain of a checkpoint, oldest first."""
        by_path = {record.path: record for record in self.list_checkpoints()}
        chain: list[CheckpointRecord] = []
        current = by_path.get(str(checkpoint_path))
        seen: set[str] = set()
        while current is not None and current.path not in seen:
            chain.append(current)
            seen.add(current.path)
            current = by_path.get(current.parent) if current.parent else None
        return list(reversed(chain))

    # -- evals -----------------------------------------------------------------

    def record_eval(self, report: dict[str, Any]) -> str:
        eval_id = report.get("id") or f"eval-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        report = {**report, "id": eval_id}
        path = self._evals_dir() / f"{eval_id}.json"
        with self._lock:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
            os.replace(tmp, path)
        return eval_id

    def get_eval(self, eval_id: str) -> dict[str, Any] | None:
        path = self._evals_dir() / f"{eval_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def list_evals(self) -> list[dict[str, Any]]:
        reports = []
        for path in sorted(self._evals_dir().glob("eval-*.json")):
            try:
                reports.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                continue
        return sorted(reports, key=lambda r: r.get("created_at", 0), reverse=True)
