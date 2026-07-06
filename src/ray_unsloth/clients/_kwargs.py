from __future__ import annotations

import warnings
from typing import Any

_SEEN: set[tuple[str, tuple[str, ...]]] = set()


def warn_ignored(kwargs: dict[str, Any], *, method: str, accepted: tuple[str, ...]) -> None:
    unknown = tuple(sorted(key for key, value in kwargs.items() if value is not None))
    if not unknown:
        return
    key = (method, unknown)
    if key in _SEEN:
        return
    _SEEN.add(key)
    warnings.warn(
        f"{method} ignored {', '.join(unknown)}; accepted parameters: {', '.join(accepted) or 'none'}.",
        UserWarning,
        stacklevel=2,
    )
