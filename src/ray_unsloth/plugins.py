"""Typed plugin registries.

Every extensible surface in ray-unsloth (runtime providers, losses, eval
scorers, dataset adapters, exporters, ...) is backed by a :class:`Registry`.
A registry maps a string name to either a live object or a lazy reference
(``"package.module:attribute"``) that is imported on first use.

Third-party packages can contribute plugins two ways:

1. **Entry points** — declare an entry point in the ``ray_unsloth.plugins``
   group. The entry point must resolve to a callable taking no arguments;
   it is invoked once and is expected to register into the registries it
   extends (imported lazily via :func:`load_entry_point_plugins`).

2. **Config** — list module paths under ``plugins:`` in a runtime config.
   Each module is imported by :func:`load_config_plugins`; import side
   effects (or a module-level ``register()`` function, if present) perform
   the registration.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from ray_unsloth.errors import PluginError

T = TypeVar("T")

ENTRY_POINT_GROUP = "ray_unsloth.plugins"


@dataclass(slots=True)
class _LazyRef:
    """A ``module:attribute`` reference resolved on first access."""

    target: str

    def resolve(self) -> Any:
        module_name, _, attr = self.target.partition(":")
        if not module_name or not attr:
            raise PluginError(
                f"Invalid lazy plugin reference '{self.target}'. Expected the form 'package.module:attribute'."
            )
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise PluginError(
                f"Could not import module '{module_name}' for plugin reference '{self.target}': {exc}"
            ) from exc
        try:
            return getattr(module, attr)
        except AttributeError as exc:
            raise PluginError(
                f"Module '{module_name}' has no attribute '{attr}' (from plugin reference '{self.target}')."
            ) from exc


@dataclass
class Registry(Generic[T]):
    """A named collection of plugins of a common kind.

    ``kind`` is a human-readable noun ("runtime provider", "loss function")
    used in error messages. Entries may be concrete objects or lazy
    ``module:attr`` strings registered via :meth:`register_lazy`.
    """

    kind: str
    _entries: dict[str, Any] = field(default_factory=dict)
    _descriptions: dict[str, str] = field(default_factory=dict)

    def register(
        self, name: str, value: T | None = None, *, description: str = "", replace: bool = False
    ) -> Callable[[T], T] | T:
        """Register ``value`` under ``name``. Usable as a decorator when ``value`` is omitted."""

        def _add(item: T) -> T:
            if not replace and name in self._entries:
                raise PluginError(
                    f"A {self.kind} named '{name}' is already registered. Pass replace=True to override it."
                )
            self._entries[name] = item
            if description:
                self._descriptions[name] = description
            return item

        if value is None:
            return _add
        return _add(value)

    def register_lazy(self, name: str, target: str, *, description: str = "", replace: bool = False) -> None:
        """Register a ``'package.module:attribute'`` reference imported on first :meth:`get`."""
        self.register(name, _LazyRef(target), description=description, replace=replace)  # type: ignore[arg-type]

    def get(self, name: str) -> T:
        try:
            entry = self._entries[name]
        except KeyError:
            raise PluginError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(sorted(self._entries)) or 'none registered'}."
            ) from None
        if isinstance(entry, _LazyRef):
            entry = entry.resolve()
            self._entries[name] = entry
        return cast(T, entry)

    def unregister(self, name: str) -> None:
        self._entries.pop(name, None)
        self._descriptions.pop(name, None)

    def describe(self, name: str) -> str:
        return self._descriptions.get(name, "")

    def names(self) -> list[str]:
        return sorted(self._entries)

    def items(self) -> Iterator[tuple[str, T]]:
        for name in self.names():
            yield name, self.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Plugin loading
# ---------------------------------------------------------------------------

_entry_points_loaded = False


def load_entry_point_plugins(*, force: bool = False) -> list[str]:
    """Invoke every callable in the ``ray_unsloth.plugins`` entry-point group once.

    Returns the names of the entry points that were loaded. Errors in a single
    plugin are wrapped in :class:`PluginError` naming the offending entry point
    rather than silently skipped.
    """
    global _entry_points_loaded
    if _entry_points_loaded and not force:
        return []
    from importlib import metadata

    loaded: list[str] = []
    for entry_point in metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            hook = entry_point.load()
            if callable(hook):
                hook()
            loaded.append(entry_point.name)
        except Exception as exc:
            raise PluginError(
                f"Plugin entry point '{entry_point.name}' ({entry_point.value}) failed to load: {exc}"
            ) from exc
    _entry_points_loaded = True
    return loaded


def load_config_plugins(modules: list[str]) -> list[str]:
    """Import each module listed under ``plugins:`` in a runtime config.

    If the imported module defines a top-level ``register()`` callable it is
    invoked; otherwise import side effects are assumed to have registered
    everything.
    """
    loaded: list[str] = []
    for module_path in modules:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise PluginError(
                f"Could not import plugin module '{module_path}' listed in the "
                f"config 'plugins' section: {exc}. Check that the package is "
                "installed and the module path is spelled correctly."
            ) from exc
        register = getattr(module, "register", None)
        if callable(register):
            register()
        loaded.append(module_path)
    return loaded


# ---------------------------------------------------------------------------
# Built-in registries. Concrete built-ins are attached lazily by the modules
# that define them (providers, losses, evals, export) to avoid import cycles;
# see each module's ``_register_builtins`` block.
# ---------------------------------------------------------------------------

providers: Registry[Any] = Registry(kind="runtime provider")
losses: Registry[Any] = Registry(kind="loss function")
scorers: Registry[Any] = Registry(kind="eval scorer")
dataset_loaders: Registry[Any] = Registry(kind="dataset loader")
exporters: Registry[Any] = Registry(kind="exporter")
checkpoint_stores: Registry[Any] = Registry(kind="checkpoint store")
metric_loggers: Registry[Any] = Registry(kind="metric logger")
recipes: Registry[Any] = Registry(kind="model recipe")


def all_registries() -> dict[str, Registry[Any]]:
    return {
        "providers": providers,
        "losses": losses,
        "scorers": scorers,
        "dataset_loaders": dataset_loaders,
        "exporters": exporters,
        "checkpoint_stores": checkpoint_stores,
        "metric_loggers": metric_loggers,
        "recipes": recipes,
    }
