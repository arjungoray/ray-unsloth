"""Dataset loaders and example-config helpers for gallery scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ray_unsloth.evals import load_dataset


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""

    rows: list[dict[str, Any]] = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(dict(json.loads(line)))
    return rows


def load_examples_block(config: str | Path | dict[str, Any], name: str) -> dict[str, Any]:
    """Load the ``examples[name]`` block from a YAML file or mapping."""

    if isinstance(config, (str, Path)):
        with Path(config).expanduser().open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    else:
        loaded = dict(config)
    examples = loaded.get("examples", {})
    if not isinstance(examples, dict):
        return {}
    block = examples.get(name, {})
    return dict(block) if isinstance(block, dict) else {}


__all__ = [
    "load_dataset",
    "load_examples_block",
    "load_jsonl",
]
