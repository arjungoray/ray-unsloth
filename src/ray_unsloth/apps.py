"""Application registry and manifest definitions."""

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, cast

from ray_unsloth.plugins import apps as _registry


@dataclass(slots=True)
class StageSpec:
    name: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AppManifest:
    name: str
    description: str
    stages: list[StageSpec]
    build_cli: Callable[[argparse._SubParsersAction[Any]], None]
    api_router: Callable[[], Any] | None = None
    requires: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "stages": [stage.to_dict() for stage in self.stages],
            "requires": list(self.requires),
        }


def register_app(manifest: AppManifest, *, replace: bool = False) -> AppManifest:
    if not replace and manifest.name in _registry:
        return cast(AppManifest, _registry.get(manifest.name))
    _registry.register(manifest.name, manifest, description=manifest.description, replace=replace)
    return manifest


def get_app(name: str) -> AppManifest:
    return cast(AppManifest, _registry.get(name))


def list_apps() -> list[AppManifest]:
    return [manifest for _name, manifest in _registry.items()]


def _requirement_installed(requirement: str) -> bool:
    package = requirement.strip()
    for separator in ("[", ">", "<", "="):
        if separator in package:
            package = package.split(separator, 1)[0]
    package = package.strip()
    return bool(package) and importlib.util.find_spec(package) is not None


def app_install_status(manifest: AppManifest) -> dict[str, Any]:
    missing = [requirement for requirement in manifest.requires if not _requirement_installed(requirement)]
    return {
        "requires": list(manifest.requires),
        "installed": not missing,
        "missing_requires": missing,
    }


if "scribe" not in _registry:
    _registry.register_lazy("scribe", "ray_unsloth_apps.scribe:MANIFEST", description="Personal writing-voice app")
