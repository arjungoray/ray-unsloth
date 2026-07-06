"""Scorer registry for lightweight eval workflows."""

from __future__ import annotations

import re
from typing import Any, Callable

from ray_unsloth.plugins import scorers as _registry

Scorer = Callable[[str, dict[str, Any]], float]


def register_scorer(name: str, scorer: Scorer, *, description: str = "", replace: bool = False) -> Scorer:
    _registry.register(name, scorer, description=description, replace=replace)
    return scorer


def get_scorer(name: str) -> Scorer:
    return _registry.get(name)


def list_scorers() -> list[str]:
    return _registry.names()


def _expected(item: dict[str, Any]) -> str:
    return str(item.get("expected", item.get("answer", "")))


def _exact_match(text: str, item: dict[str, Any]) -> float:
    return 1.0 if text.strip() == _expected(item).strip() else 0.0


def _contains(text: str, item: dict[str, Any]) -> float:
    expected = _expected(item).strip()
    return 1.0 if expected and expected in text else 0.0


def _regex(text: str, item: dict[str, Any]) -> float:
    pattern = str(item.get("pattern", item.get("expected", "")))
    return 1.0 if pattern and re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) else 0.0


def _numeric_match(text: str, item: dict[str, Any]) -> float:
    expected_raw = item.get("expected", item.get("answer"))
    try:
        expected = float(expected_raw)
    except (TypeError, ValueError):
        return 0.0
    tolerance = float(item.get("tolerance", 1e-6))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    for value in numbers:
        if abs(float(value) - expected) <= tolerance:
            return 1.0
    return 0.0


for _name, _fn, _description in (
    ("exact_match", _exact_match, "1.0 when stripped text exactly matches expected/answer."),
    ("contains", _contains, "1.0 when generated text contains expected/answer."),
    ("regex", _regex, "1.0 when generated text matches item.pattern or expected as regex."),
    ("numeric_match", _numeric_match, "1.0 when any number in generated text matches expected within tolerance."),
):
    if _name not in _registry:
        register_scorer(_name, _fn, description=_description)
