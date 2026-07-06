"""Client-side run recording.

Observability lives at the client, not the engine: every provider (Ray,
Modal, fake, future clusters) gets metrics, logs, and checkpoint lineage for
free because recording wraps the futures the ``TrainingClient`` returns and
writes to the local :class:`~ray_unsloth.store.RunStore` when they resolve.

Recording is best-effort by design — a failure to write a metric never
breaks a training step.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from ray_unsloth.store import RunStore


class RecordingFuture:
    """Wraps any ray-unsloth future; invokes ``callback(value)`` once on resolve."""

    def __init__(self, inner: Any, callback: Callable[[Any], None]):
        self._inner = inner
        self._callback = callback
        self._recorded = False
        self._lock = threading.Lock()

    def _record(self, value: Any) -> Any:
        with self._lock:
            if not self._recorded:
                self._recorded = True
                try:
                    self._callback(value)
                except Exception:
                    pass
        return value

    def result(self, timeout: float | None = None) -> Any:
        return self._record(self._inner.result(timeout))

    async def result_async(self, timeout: float | None = None) -> Any:
        result_async = getattr(self._inner, "result_async", None)
        if callable(result_async):
            return self._record(await result_async(timeout))
        return self.result(timeout)

    def get(self, timeout: float | None = None) -> Any:
        getter = getattr(self._inner, "get", None)
        if callable(getter):
            return self._record(getter(timeout))
        return self.result(timeout)

    def __await__(self):
        async def _await_inner():
            awaitable = getattr(self._inner, "__await__", None)
            if not callable(awaitable):
                return self.result()
            value = await self._inner
            # AsyncMethodFuture.__await__ submits work and returns itself; keep
            # that two-stage contract so callers can still call result_async().
            if value is self._inner:
                return self
            return self._record(value)

        return _await_inner().__await__()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class RunRecorder:
    """Streams one training client's events into the run store."""

    def __init__(self, store: RunStore, run_id: str, *, base_model: str | None = None):
        self.store = store
        self.run_id = run_id
        self.base_model = base_model
        self._last_checkpoint_path: str | None = None
        self._lock = threading.Lock()

    # -- wrapping ---------------------------------------------------------------

    def wrap_forward_backward(self, future: Any, *, loss_fn: str) -> RecordingFuture:
        def _on_result(output: Any) -> None:
            metrics = {"loss": getattr(output, "loss", None), "loss_fn": loss_fn}
            extra = getattr(output, "metrics", None)
            if isinstance(extra, dict):
                for key, value in extra.items():
                    if isinstance(value, (int, float)) and key not in ("loss",):
                        metrics[f"train/{key}"] = value
            self.store.append_metrics(self.run_id, {"event": "forward_backward", **metrics})

        return RecordingFuture(future, _on_result)

    def wrap_optim_step(self, future: Any) -> RecordingFuture:
        def _on_result(output: Any) -> None:
            step = getattr(output, "step", None)
            self.store.append_metrics(self.run_id, {"event": "optim_step", "step": step})

        return RecordingFuture(future, _on_result)

    def wrap_save(self, future: Any, *, kind: str, has_optimizer: bool = False) -> RecordingFuture:
        def _on_result(output: Any) -> None:
            path = getattr(output, "path", None)
            if path is None:
                checkpoint = getattr(output, "checkpoint", None)
                path = getattr(checkpoint, "path", None)
            if path is None:
                return
            step = getattr(output, "step", None)
            with self._lock:
                parent = self._last_checkpoint_path
                self.store.record_checkpoint(
                    path=str(path),
                    run_id=self.run_id,
                    step=step,
                    kind=kind,
                    parent=parent,
                    has_optimizer=has_optimizer,
                    base_model=self.base_model,
                )
                if kind == "training_state":
                    self._last_checkpoint_path = str(path)
            self.store.append_log(self.run_id, f"saved {kind} checkpoint: {path}")

        return RecordingFuture(future, _on_result)

    def note_loaded_checkpoint(self, path: str) -> None:
        """Called when a client restores from a checkpoint — sets lineage parent."""
        with self._lock:
            self._last_checkpoint_path = str(path)
        self.store.append_log(self.run_id, f"restored from checkpoint: {path}")

    def log(self, message: str, *, level: str = "info") -> None:
        self.store.append_log(self.run_id, message, level=level)

    def finish(self, status: str = "completed") -> None:
        self.store.update_run(self.run_id, status=status)
