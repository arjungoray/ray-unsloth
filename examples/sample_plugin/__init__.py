"""Example ray-unsloth plugin.

Use from config:

plugins:
  - examples.sample_plugin

It registers:
- an eval scorer named ``length_bonus``
- a local export target named ``metadata-card``
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ray_unsloth.checkpoints import read_manifest, resolve_path
from ray_unsloth.evals import register_scorer
from ray_unsloth.export import ExportReport, register_exporter


def _length_bonus(text: str, item: dict[str, Any]) -> float:
    minimum = int(item.get("min_chars", 1))
    return min(1.0, len(text) / max(1, minimum))


class MetadataCardExporter:
    name = "metadata-card"

    def export(self, checkpoint_path: str, output: str | None = None, **options: Any) -> ExportReport:
        del options
        source = resolve_path(checkpoint_path)
        manifest = read_manifest(source)
        out = resolve_path(output or f"{source}-metadata-card")
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_path": str(source),
            "base_model": manifest.get("base_model"),
            "step": manifest.get("step"),
            "lora": manifest.get("lora", {}),
            "created_at": time.time(),
        }
        (out / "metadata-card.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        return ExportReport(
            target=self.name,
            source_path=str(source),
            output_path=str(out),
            created_at=payload["created_at"],
            artifacts=["metadata-card.json"],
            notes=["Example plugin exporter wrote checkpoint metadata."],
        )


def register() -> None:
    register_scorer("length_bonus", _length_bonus, description="Score by generated text length.", replace=True)
    register_exporter(
        "metadata-card",
        MetadataCardExporter(),
        description="Example plugin exporter that writes a metadata card.",
        replace=True,
    )


register()
