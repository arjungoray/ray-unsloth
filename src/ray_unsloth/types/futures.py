from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")

_WARNED_PROXY_GETATTR = False


class RayObjectFuture(Generic[T]):
    """Small future wrapper around a Ray ObjectRef."""

    def __init__(self, object_ref: Any):
        self.object_ref = object_ref

    def result(self, timeout: float | None = None) -> T:
        try:
            import ray
        except ImportError as exc:  # pragma: no cover - exercised without deps
            raise RuntimeError("Ray is required to resolve this future.") from exc
        if timeout is None:
            return ray.get(self.object_ref)
        return ray.get(self.object_ref, timeout=timeout)

    async def result_async(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def get(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def __await__(self):
        async def _resolve():
            return self.result()

        return _resolve().__await__()


class ImmediateFuture(Generic[T]):
    """Future wrapper for synchronous local values."""

    def __init__(self, value: T):
        self._value = value

    def result(self, timeout: float | None = None) -> T:
        del timeout
        return self._value

    async def result_async(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def get(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def __await__(self):
        async def _resolve():
            return self.result()

        return _resolve().__await__()


class FutureValueProxy(Generic[T]):
    """Resolved value that still accepts `.result()` for older local examples."""

    def __init__(self, value: T):
        object.__setattr__(self, "_value", value)

    def result(self, timeout: float | None = None) -> T:
        del timeout
        return object.__getattribute__(self, "_value")

    async def result_async(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def get(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def __await__(self):
        async def _resolve():
            return self.result()

        return _resolve().__await__()

    def __getattr__(self, name: str) -> Any:
        global _WARNED_PROXY_GETATTR
        if not _WARNED_PROXY_GETATTR:
            warnings.warn(
                "FutureValueProxy attribute access is deprecated; call .result() first.",
                DeprecationWarning,
                stacklevel=2,
            )
            _WARNED_PROXY_GETATTR = True
        return getattr(self.result(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.result(), name, value)

    def __call__(self, *args, **kwargs):
        return self.result()(*args, **kwargs)

    def __iter__(self):
        return iter(self.result())

    def __len__(self) -> int:
        return len(self.result())  # type: ignore[arg-type]

    def __getitem__(self, key):
        return self.result()[key]  # type: ignore[index]

    def __bool__(self) -> bool:
        return bool(self.result())

    def __str__(self) -> str:
        return str(self.result())

    def __repr__(self) -> str:
        return repr(self.result())

    def __eq__(self, other: Any) -> bool:
        return self.result() == other


def future_from(value: Any) -> RayObjectFuture[Any] | ImmediateFuture[Any]:
    """Wrap Ray object refs and synchronous values behind one future protocol."""

    if hasattr(value, "hex") and value.__class__.__name__ == "ObjectRef":
        return RayObjectFuture(value)
    return ImmediateFuture(value)


class AsyncMethodFuture(Generic[T]):
    """Future returned by async training client methods before completion."""

    def __init__(
        self,
        future: RayObjectFuture[T] | ImmediateFuture[T] | None = None,
        *,
        submit_sync: Callable[[], RayObjectFuture[T] | ImmediateFuture[T]] | None = None,
        submit_async: Callable[[], Awaitable[Any]] | None = None,
    ):
        self._future = future
        self._submit_sync = submit_sync
        self._submit_async = submit_async

    def _ensure_sync_future(self) -> RayObjectFuture[T] | ImmediateFuture[T]:
        if self._future is None:
            if self._submit_sync is None:
                raise RuntimeError("This async future can only be resolved from an async context.")
            self._future = self._submit_sync()
        return self._future

    async def _ensure_async_future(self) -> RayObjectFuture[T] | ImmediateFuture[T]:
        if self._future is None:
            if self._submit_async is None:
                return self._ensure_sync_future()
            self._future = future_from(await self._submit_async())
        return self._future

    def result(self, timeout: float | None = None) -> T:
        return self._ensure_sync_future().result(timeout=timeout)

    async def result_async(self, timeout: float | None = None) -> T:
        return await (await self._ensure_async_future()).result_async(timeout=timeout)

    def get(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def __await__(self):
        async def _submitted():
            await self._ensure_async_future()
            return self

        return _submitted().__await__()


def async_method_future(value: RayObjectFuture[T] | ImmediateFuture[T]) -> AsyncMethodFuture[T]:
    return AsyncMethodFuture(value)


__all__ = [
    "AsyncMethodFuture",
    "FutureValueProxy",
    "ImmediateFuture",
    "RayObjectFuture",
    "async_method_future",
    "future_from",
]
