"""Helpers for calling either Ray actors or local test doubles."""

from __future__ import annotations

from typing import Any

from ray_unsloth.types import future_from


def call(actor: Any, method_name: str, *args, **kwargs):
    method = getattr(actor, method_name)
    remote = getattr(method, "remote", None)
    if callable(remote):
        return future_from(remote(*args, **kwargs))
    return future_from(method(*args, **kwargs))


def resolve(value: Any) -> Any:
    result = getattr(value, "result", None)
    if callable(result):
        return result()
    get = getattr(value, "get", None)
    if callable(get):
        return get()
    return value
