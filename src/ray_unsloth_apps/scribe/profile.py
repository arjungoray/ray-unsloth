"""Stylometry and style-profile helpers for Scribe."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from ray_unsloth_apps.scribe.ingest import Passage

FUNCTION_WORDS = [
    "the",
    "of",
    "and",
    "a",
    "to",
    "in",
    "is",
    "that",
    "it",
    "for",
    "on",
    "with",
    "as",
    "but",
    "at",
    "by",
    "this",
    "so",
    "just",
    "really",
]

FEATURE_NAMES = [
    "sentence_length_mean",
    "sentence_length_std",
    "comma_rate_per_1000",
    "semicolon_rate_per_1000",
    "emdash_rate_per_1000",
    "exclamation_rate_per_1000",
    "question_rate_per_1000",
    "paren_rate_per_1000",
    "contraction_rate_per_100_words",
    "type_token_ratio",
    "mean_word_length",
    "uppercase_word_rate_per_100_words",
    "digit_rate_per_1000",
    "newline_rate_per_1000",
    "paragraph_rate_per_1000",
    *[f"func_{word}_per_100_words" for word in FUNCTION_WORDS],
]

_WORD_RE = re.compile(r"\b[\w']+\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CONTRACTION_RE = re.compile(r"\b\w+(?:n't|'re|'s|'ll|'ve|'d)\b", re.IGNORECASE)
_END_PUNCT_RE = re.compile(r"[.!?]+")
_SHINGLE_SIZE = 8


@dataclass(slots=True)
class StyleProfile:
    features_mean: dict[str, float]
    features_std: dict[str, float]
    ngram_hashes: set[int]
    n_passages: int
    n_words: int

    def to_dict(self) -> dict[str, object]:
        return {
            "features_mean": dict(self.features_mean),
            "features_std": dict(self.features_std),
            "ngram_hashes": sorted(self.ngram_hashes),
            "n_passages": self.n_passages,
            "n_words": self.n_words,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> StyleProfile:
        return cls(
            features_mean={str(key): float(value) for key, value in dict(data["features_mean"]).items()},
            features_std={str(key): float(value) for key, value in dict(data["features_std"]).items()},
            ngram_hashes={int(value) for value in list(data.get("ngram_hashes", []))},
            n_passages=int(data.get("n_passages", 0)),
            n_words=int(data.get("n_words", 0)),
        )


def stylometrics(text: str) -> dict[str, float]:
    words = _WORD_RE.findall(text)
    lower_words = [word.lower() for word in words]
    sentences = _split_sentences(text)
    sentence_lengths = [len(_WORD_RE.findall(sentence)) for sentence in sentences if _WORD_RE.findall(sentence)] or [
        len(words)
    ]
    char_count = max(1, len(text))
    word_count = max(1, len(words))
    first_200 = lower_words[:200]
    first_200_count = max(1, len(first_200))
    unique_first_200 = len(set(first_200))
    paragraph_count = max(1, len([chunk for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]))

    features: dict[str, float] = {
        "sentence_length_mean": float(mean(sentence_lengths)),
        "sentence_length_std": float(_std(sentence_lengths)),
        "comma_rate_per_1000": _rate(text.count(","), char_count),
        "semicolon_rate_per_1000": _rate(text.count(";"), char_count),
        "emdash_rate_per_1000": _rate(text.count("—") + text.count("--"), char_count),
        "exclamation_rate_per_1000": _rate(text.count("!"), char_count),
        "question_rate_per_1000": _rate(text.count("?"), char_count),
        "paren_rate_per_1000": _rate(text.count("(") + text.count(")"), char_count),
        "contraction_rate_per_100_words": _rate(len(_CONTRACTION_RE.findall(text)), word_count),
        "type_token_ratio": float(unique_first_200 / first_200_count),
        "mean_word_length": float(sum(len(word) for word in words) / word_count),
        "uppercase_word_rate_per_100_words": _rate(
            sum(1 for word in words if word.isupper() and len(word) > 1), word_count
        ),
        "digit_rate_per_1000": _rate(sum(ch.isdigit() for ch in text), char_count),
        "newline_rate_per_1000": _rate(text.count("\n"), char_count),
        "paragraph_rate_per_1000": _rate(max(0, paragraph_count - 1), char_count),
    }
    total_words = max(1, len(words))
    for word in FUNCTION_WORDS:
        features[f"func_{word}_per_100_words"] = _rate(sum(1 for item in lower_words if item == word), total_words)
    return features


def build_profile(passages: Iterable[Passage | str]) -> StyleProfile:
    texts = [_passage_text(item) for item in passages]
    if not texts:
        return StyleProfile(
            features_mean={name: 0.0 for name in FEATURE_NAMES},
            features_std={name: 1.0 for name in FEATURE_NAMES},
            ngram_hashes=set(),
            n_passages=0,
            n_words=0,
        )
    feature_rows = [stylometrics(text) for text in texts]
    features_mean = {name: float(sum(row[name] for row in feature_rows) / len(feature_rows)) for name in FEATURE_NAMES}
    features_std = {name: float(max(_std([row[name] for row in feature_rows]), 1e-3)) for name in FEATURE_NAMES}
    ngram_hashes = set().union(*(_shingle_hashes(text) for text in texts))
    n_words = sum(len(_WORD_RE.findall(text)) for text in texts)
    return StyleProfile(
        features_mean=features_mean,
        features_std=features_std,
        ngram_hashes=ngram_hashes,
        n_passages=len(texts),
        n_words=n_words,
    )


def save_profile(profile: StyleProfile, path: str | Path) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_profile(path: str | Path) -> StyleProfile:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return StyleProfile.from_dict(data)


def stylometry_distance(text: str, profile: StyleProfile) -> float:
    features = stylometrics(text)
    distances = []
    for name in FEATURE_NAMES:
        std = max(profile.features_std.get(name, 1e-3), 1e-3)
        mean_value = profile.features_mean.get(name, 0.0)
        z = abs((features[name] - mean_value) / std)
        distances.append(min(3.0, z))
    return min(3.0, float(sum(distances) / max(1, len(distances))))


def copy_overlap(text: str, profile: StyleProfile) -> float:
    shingles = _shingle_hashes(text)
    if not shingles:
        return 0.0
    matches = sum(1 for value in shingles if value in profile.ngram_hashes)
    return matches / len(shingles)


def capability_report(profile: StyleProfile) -> str:
    if profile.n_words < 10_000:
        return f"starter: {profile.n_words} words across {profile.n_passages} passages; add more text to stabilize the voice model."
    if profile.n_words <= 50_000:
        return f"solid: {profile.n_words} words across {profile.n_passages} passages; the model should capture the main style, but more breadth still helps."
    return f"rich: {profile.n_words} words across {profile.n_passages} passages; you have enough text for a sharper and more faithful style model."


def feature_vector(text: str) -> list[float]:
    metrics = stylometrics(text)
    return [metrics[name] for name in FEATURE_NAMES]


def _passage_text(item: Passage | str) -> str:
    return item.text if isinstance(item, Passage) else str(item)


def _split_sentences(text: str) -> list[str]:
    sentences = [chunk.strip() for chunk in _SENTENCE_SPLIT_RE.split(text.strip()) if chunk.strip()]
    return sentences or [text.strip()]


def _rate(count: float, denominator: float) -> float:
    return float(count) / max(1.0, float(denominator)) * 100.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _shingle_hashes(text: str) -> set[int]:
    words = [word.lower() for word in _WORD_RE.findall(text)]
    if len(words) < _SHINGLE_SIZE:
        return set()
    hashes: set[int] = set()
    for start in range(0, len(words) - _SHINGLE_SIZE + 1):
        shingle = " ".join(words[start : start + _SHINGLE_SIZE])
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        hashes.add(int.from_bytes(digest, "big"))
    return hashes
