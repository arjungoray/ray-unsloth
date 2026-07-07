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
from typing import cast

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
    "readability_flesch_ease",
    "readability_fk_grade",
    "register_first_person_rate",
    "register_second_person_rate",
    "register_contraction_rate",
    "register_passive_proxy",
    "register_hedging_rate",
    "tone_exclamation_rate",
    "tone_question_rate",
    "tone_intensifier_rate",
    "tone_parenthetical_rate",
    "discourse_transition_rate",
    "discourse_example_rate",
    "discourse_contrast_rate",
    "discourse_opener_conjunction_rate",
    "discourse_bullet_rate",
    "cognitive_concrete_example_density",
    "cognitive_sentence_len_cv",
    "cognitive_imperative_rate",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "stylometry": [
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
    ],
    "readability": [
        "readability_flesch_ease",
        "readability_fk_grade",
    ],
    "register": [
        "register_first_person_rate",
        "register_second_person_rate",
        "register_contraction_rate",
        "register_passive_proxy",
        "register_hedging_rate",
    ],
    "tone": [
        "tone_exclamation_rate",
        "tone_question_rate",
        "tone_intensifier_rate",
        "tone_parenthetical_rate",
    ],
    "discourse": [
        "discourse_transition_rate",
        "discourse_example_rate",
        "discourse_contrast_rate",
        "discourse_opener_conjunction_rate",
        "discourse_bullet_rate",
    ],
    "cognitive": [
        "cognitive_concrete_example_density",
        "cognitive_sentence_len_cv",
        "cognitive_imperative_rate",
    ],
}

REFERENCE_NORMS: dict[str, tuple[float, float]] = {
    "sentence_length_mean": (18.0, 6.0),
    "sentence_length_std": (7.0, 3.0),
    "comma_rate_per_1000": (12.0, 8.0),
    "semicolon_rate_per_1000": (1.5, 2.5),
    "emdash_rate_per_1000": (0.8, 1.5),
    "exclamation_rate_per_1000": (1.0, 2.0),
    "question_rate_per_1000": (1.5, 2.0),
    "paren_rate_per_1000": (1.5, 2.5),
    "contraction_rate_per_100_words": (10.0, 8.0),
    "type_token_ratio": (0.55, 0.10),
    "mean_word_length": (4.8, 0.5),
    "uppercase_word_rate_per_100_words": (0.2, 0.5),
    "digit_rate_per_1000": (4.0, 6.0),
    "newline_rate_per_1000": (2.0, 3.0),
    "paragraph_rate_per_1000": (0.8, 1.2),
    "func_the_per_100_words": (5.0, 2.0),
    "func_of_per_100_words": (3.5, 1.8),
    "func_and_per_100_words": (3.0, 1.8),
    "func_a_per_100_words": (3.5, 1.8),
    "func_to_per_100_words": (4.5, 2.0),
    "func_in_per_100_words": (4.0, 1.8),
    "func_is_per_100_words": (3.0, 1.5),
    "func_that_per_100_words": (2.5, 1.5),
    "func_it_per_100_words": (2.8, 1.5),
    "func_for_per_100_words": (1.8, 1.2),
    "func_on_per_100_words": (1.8, 1.2),
    "func_with_per_100_words": (1.6, 1.0),
    "func_as_per_100_words": (1.2, 0.9),
    "func_but_per_100_words": (0.9, 0.8),
    "func_at_per_100_words": (1.0, 0.8),
    "func_by_per_100_words": (0.8, 0.8),
    "func_this_per_100_words": (1.0, 0.8),
    "func_so_per_100_words": (1.2, 0.9),
    "func_just_per_100_words": (0.5, 0.6),
    "func_really_per_100_words": (0.4, 0.5),
    "readability_flesch_ease": (55.0, 15.0),
    "readability_fk_grade": (10.0, 4.0),
    "register_first_person_rate": (3.0, 3.0),
    "register_second_person_rate": (1.5, 2.5),
    "register_contraction_rate": (10.0, 8.0),
    "register_passive_proxy": (15.0, 10.0),
    "register_hedging_rate": (2.5, 2.5),
    "tone_exclamation_rate": (1.0, 1.5),
    "tone_question_rate": (1.5, 2.0),
    "tone_intensifier_rate": (3.0, 3.0),
    "tone_parenthetical_rate": (1.5, 2.0),
    "discourse_transition_rate": (6.0, 4.0),
    "discourse_example_rate": (1.2, 1.5),
    "discourse_contrast_rate": (3.0, 2.5),
    "discourse_opener_conjunction_rate": (8.0, 6.0),
    "discourse_bullet_rate": (3.0, 5.0),
    "cognitive_concrete_example_density": (5.0, 4.0),
    "cognitive_sentence_len_cv": (0.35, 0.18),
    "cognitive_imperative_rate": (1.0, 2.0),
}

_WORD_RE = re.compile(r"\b[\w']+\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CONTRACTION_RE = re.compile(r"\b\w+(?:n't|'re|'s|'ll|'ve|'d)\b", re.IGNORECASE)
_END_PUNCT_RE = re.compile(r"[.!?]+")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)
_REGISTER_FIRST_PERSON = {"i", "me", "my", "we", "our"}
_REGISTER_SECOND_PERSON = {"you", "your"}
_REGISTER_HEDGES = {"maybe", "perhaps", "likely", "probably", "might", "could"}
_TONE_INTENSIFIERS = {"very", "really", "extremely", "incredibly", "genuinely", "absolutely"}
_DISCOURSE_TRANSITIONS = {"however", "therefore", "because", "so", "then", "instead", "meanwhile", "also"}
_DISCOURSE_CONTRASTS = {"but", "though", "although", "yet", "whereas"}
_DISCOURSE_OPENERS = {"and", "but", "so", "or", "because"}
_COGNITIVE_IMPERATIVES = {
    "make",
    "run",
    "keep",
    "use",
    "add",
    "write",
    "check",
    "say",
    "stop",
    "start",
    "note",
    "remember",
    "lead",
    "report",
    "ship",
    "fix",
    "test",
    "delete",
    "avoid",
    "bias",
}
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
        features_mean = cast(dict[str, object], data.get("features_mean", {}))
        features_std = cast(dict[str, object], data.get("features_std", {}))
        return cls(
            features_mean=_normalize_feature_map(features_mean, default=0.0),
            features_std=_normalize_feature_map(features_std, default=1.0),
            ngram_hashes={int(value) for value in list(data.get("ngram_hashes", []))},
            n_passages=int(data.get("n_passages", 0)),
            n_words=int(data.get("n_words", 0)),
        )


def stylometrics(text: str) -> dict[str, float]:
    words = _WORD_RE.findall(text)
    lower_words = [word.lower() for word in words]
    sentences = _split_sentences(text)
    sentence_word_lists = [_WORD_RE.findall(sentence) for sentence in sentences]
    sentence_lengths = [len(sentence_words) for sentence_words in sentence_word_lists if sentence_words] or [len(words)]
    char_count = max(1, len(text))
    word_count = max(1, len(words))
    sentence_count = max(1, len(sentences))
    line_count = max(1, len([line for line in text.splitlines() if line.strip()]))
    syllable_count = sum(_syllable_count(word) for word in words) or 1
    first_200 = lower_words[:200]
    first_200_count = max(1, len(first_200))
    unique_first_200 = len(set(first_200))
    paragraph_count = max(1, len([chunk for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]))
    passive_proxy_count = sum(_has_passive_proxy(sentence_words) for sentence_words in sentence_word_lists)
    first_person_rate = _rate(sum(1 for item in lower_words if item in _REGISTER_FIRST_PERSON), word_count)
    second_person_rate = _rate(sum(1 for item in lower_words if item in _REGISTER_SECOND_PERSON), word_count)
    contraction_rate = _rate(len(_CONTRACTION_RE.findall(text)), word_count)
    hedging_rate = _rate(_count_hedges(text), word_count)
    intensifier_rate = _rate(sum(1 for item in lower_words if item in _TONE_INTENSIFIERS), word_count)
    transition_rate = _rate(sum(1 for item in lower_words if item in _DISCOURSE_TRANSITIONS), word_count)
    example_rate = _rate(_count_examples(text), word_count)
    contrast_rate = _rate(
        sum(1 for item in lower_words if item in _DISCOURSE_CONTRASTS) + text.lower().count("not "),
        word_count,
    )
    opener_conjunction_rate = _rate(
        sum(1 for sentence in sentences if _first_word(sentence) in _DISCOURSE_OPENERS),
        sentence_count,
    )
    bullet_rate = _rate(_count_bullet_lines(text), line_count)
    concrete_density = _rate(sum(ch.isdigit() for ch in text) + text.count("%") + text.count("$"), word_count)
    sentence_cv = float(_std(sentence_lengths) / max(1.0, mean(sentence_lengths))) if sentence_lengths else 0.0
    imperative_rate = _rate(
        sum(1 for sentence in sentences if _first_word(sentence) in _COGNITIVE_IMPERATIVES), sentence_count
    )
    readability_flesch = float(206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count))
    readability_fk = float(0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59)

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
        "readability_flesch_ease": readability_flesch,
        "readability_fk_grade": readability_fk,
        "register_first_person_rate": first_person_rate,
        "register_second_person_rate": second_person_rate,
        "register_contraction_rate": contraction_rate,
        "register_passive_proxy": _rate(passive_proxy_count, sentence_count),
        "register_hedging_rate": hedging_rate,
        "tone_exclamation_rate": _rate(text.count("!"), char_count),
        "tone_question_rate": _rate(text.count("?"), char_count),
        "tone_intensifier_rate": intensifier_rate,
        "tone_parenthetical_rate": _rate(text.count("(") + text.count(")"), char_count),
        "discourse_transition_rate": transition_rate,
        "discourse_example_rate": example_rate,
        "discourse_contrast_rate": contrast_rate,
        "discourse_opener_conjunction_rate": opener_conjunction_rate,
        "discourse_bullet_rate": bullet_rate,
        "cognitive_concrete_example_density": concrete_density,
        "cognitive_sentence_len_cv": sentence_cv,
        "cognitive_imperative_rate": imperative_rate,
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
    common_names = [
        name
        for name in FEATURE_NAMES
        if name in features and name in profile.features_mean and name in profile.features_std
    ]
    for name in common_names:
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


def fingerprint_card(profile: StyleProfile) -> str:
    feature_ranks = _rank_profile_features(profile)
    lines = []
    for group_name, group_features in FEATURE_GROUPS.items():
        distinctive = [item for item in feature_ranks if item[0] in group_features][:3]
        if not distinctive:
            continue
        clauses = [_feature_clause(feature_name, value, z_score) for feature_name, value, z_score in distinctive]
        lines.append(f"{group_name.title()}: " + "; ".join(clauses) + ".")
    summary = _summary_clause(feature_ranks[:3])
    lines.append(f"Overall: {summary}.")
    return "\n".join(lines)


def feature_vector(text: str) -> list[float]:
    metrics = stylometrics(text)
    return [metrics[name] for name in FEATURE_NAMES]


def _passage_text(item: Passage | str) -> str:
    return item.text if isinstance(item, Passage) else str(item)


def _split_sentences(text: str) -> list[str]:
    sentences = [chunk.strip() for chunk in _SENTENCE_SPLIT_RE.split(text.strip()) if chunk.strip()]
    return sentences or [text.strip()]


def _first_word(text: str) -> str:
    match = _WORD_RE.search(text)
    return match.group(0).lower() if match else ""


def _rate(count: float, denominator: float) -> float:
    return float(count) / max(1.0, float(denominator)) * 100.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _syllable_count(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 1
    groups = _VOWEL_GROUP_RE.findall(cleaned)
    return max(1, len(groups))


def _count_hedges(text: str) -> int:
    lower_text = text.lower()
    count = 0
    for phrase in (*_REGISTER_HEDGES, "i think", "i guess"):
        count += lower_text.count(phrase)
    return count


def _count_examples(text: str) -> int:
    lower_text = text.lower()
    count = 0
    for phrase in ("for example", "for instance", "e.g.", "like a", "such as"):
        count += lower_text.count(phrase)
    return count


def _count_bullet_lines(text: str) -> int:
    count = 0
    for line in text.splitlines():
        if re.match(r"^\s*(?:-|\*|\d+\.)\s+", line):
            count += 1
    return count


def _has_passive_proxy(sentence_words: list[str]) -> int:
    auxiliaries = {"was", "were", "been", "being"}
    for index, word in enumerate(sentence_words):
        if word.lower() not in auxiliaries:
            continue
        for look_ahead in (1, 2):
            if index + look_ahead >= len(sentence_words):
                continue
            candidate = sentence_words[index + look_ahead].lower()
            if candidate.endswith("ed") or candidate.endswith("en"):
                return 1
    return 0


def _normalize_feature_map(values: dict[str, object], default: float) -> dict[str, float]:
    normalized = {str(key): float(value) for key, value in values.items()}
    for name in FEATURE_NAMES:
        normalized.setdefault(name, default)
    return normalized


def _rank_profile_features(profile: StyleProfile) -> list[tuple[str, float, float]]:
    ranked: list[tuple[str, float, float]] = []
    for feature_name in FEATURE_NAMES:
        mean_value = profile.features_mean.get(feature_name, REFERENCE_NORMS[feature_name][0])
        ref_mean, ref_std = REFERENCE_NORMS[feature_name]
        z_score = (mean_value - ref_mean) / max(ref_std, 1e-3)
        ranked.append((feature_name, mean_value, z_score))
    ranked.sort(key=lambda item: (-abs(item[2]), item[0]))
    return ranked


def _feature_clause(feature_name: str, value: float, z_score: float) -> str:
    if feature_name == "sentence_length_mean":
        return f"{_polarity_word(z_score, 'short', 'long')} sentences ({value:.1f} words avg)"
    if feature_name == "sentence_length_std":
        return f"sentence rhythm stays {_polarity_word(z_score, 'steady', 'varied')}"
    if feature_name == "comma_rate_per_1000":
        return f"{_polarity_word(z_score, 'comma-light', 'comma-heavy')}"
    if feature_name == "semicolon_rate_per_1000":
        return f"{_polarity_word(z_score, 'semicolon-light', 'semicolon-rich')}"
    if feature_name == "emdash_rate_per_1000":
        return f"{_polarity_word(z_score, 'dash-light', 'dashy')}"
    if feature_name == "exclamation_rate_per_1000":
        return f"{_polarity_word(z_score, 'restrained exclamation use', 'energetic exclamation use')}"
    if feature_name == "question_rate_per_1000":
        return f"{_polarity_word(z_score, 'declarative', 'question-led')}"
    if feature_name == "paren_rate_per_1000":
        return f"{_polarity_word(z_score, 'parenthesis-light', 'aside-heavy')}"
    if feature_name == "contraction_rate_per_100_words":
        return f"{_polarity_word(z_score, 'formal', 'conversational')}"
    if feature_name == "type_token_ratio":
        return f"{_polarity_word(z_score, 'repetitive', 'lexically varied')}"
    if feature_name == "mean_word_length":
        return f"{_polarity_word(z_score, 'plain-spoken', 'polysyllabic')}"
    if feature_name == "uppercase_word_rate_per_100_words":
        return f"{_polarity_word(z_score, 'calm', 'shouty')}"
    if feature_name == "digit_rate_per_1000":
        return f"{_polarity_word(z_score, 'text-first', 'numeric')}"
    if feature_name == "newline_rate_per_1000":
        return f"{_polarity_word(z_score, 'continuous', 'line-broken')}"
    if feature_name == "paragraph_rate_per_1000":
        return f"{_polarity_word(z_score, 'continuous paragraphs', 'sectioned')}"
    if feature_name.startswith("func_") and feature_name.endswith("_per_100_words"):
        word = feature_name.removeprefix("func_").removesuffix("_per_100_words")
        low = f"rarely uses '{word}'"
        high = f"leans on '{word}'"
        return _polarity_word(z_score, low, high)
    if feature_name == "readability_flesch_ease":
        low = f"reads like dense prose ({value:.0f} Flesch Ease)"
        high = f"reads easily ({value:.0f} Flesch Ease)"
        return _polarity_word(z_score, low, high)
    if feature_name == "readability_fk_grade":
        low = f"accessible grade level ({value:.1f})"
        high = f"higher reading grade ({value:.1f})"
        return _polarity_word(z_score, low, high)
    if feature_name == "register_first_person_rate":
        return _polarity_word(z_score, "keeps the narrator at arm's length", "frequent first-person voice")
    if feature_name == "register_second_person_rate":
        return f"{_polarity_word(z_score, 'rarely addresses the reader directly', 'direct second-person address')}"
    if feature_name == "register_contraction_rate":
        return f"{_polarity_word(z_score, 'keeps contractions rare', 'uses contractions freely')}"
    if feature_name == "register_passive_proxy":
        return f"{_polarity_word(z_score, 'stays mostly active and agentive', 'often slips into passive voice')}"
    if feature_name == "register_hedging_rate":
        return f"{_polarity_word(z_score, 'states things directly', 'hedges and qualifies often')}"
    if feature_name == "tone_exclamation_rate":
        return f"{_polarity_word(z_score, 'keeps exclamation restrained', 'punctuates with exclamation')}"
    if feature_name == "tone_question_rate":
        return f"{_polarity_word(z_score, 'remains declarative', 'asks questions often')}"
    if feature_name == "tone_intensifier_rate":
        return f"{_polarity_word(z_score, 'avoids verbal inflation', 'leans on intensifiers')}"
    if feature_name == "tone_parenthetical_rate":
        return f"{_polarity_word(z_score, 'stays parenthesis-light', 'likes parenthetical asides')}"
    if feature_name == "discourse_transition_rate":
        return f"{_polarity_word(z_score, 'links ideas implicitly', 'moves with explicit transitions')}"
    if feature_name == "discourse_example_rate":
        return f"{_polarity_word(z_score, 'stays abstract on purpose', 'illustrates with examples')}"
    if feature_name == "discourse_contrast_rate":
        return f"{_polarity_word(z_score, 'rarely pivots by contrast', 'leans into contrast and negation')}"
    if feature_name == "discourse_opener_conjunction_rate":
        return f"{_polarity_word(z_score, 'rarely opens with conjunctions', 'begins sentences with conjunctions')}"
    if feature_name == "discourse_bullet_rate":
        return f"{_polarity_word(z_score, 'prefers paragraph flow', 'often breaks into bullets')}"
    if feature_name == "cognitive_concrete_example_density":
        return f"{_polarity_word(z_score, 'stays low on concrete detail', 'packs in numbers and concrete markers')}"
    if feature_name == "cognitive_sentence_len_cv":
        return f"{_polarity_word(z_score, 'sentence rhythm is even', 'sentence rhythm varies sharply')}"
    if feature_name == "cognitive_imperative_rate":
        return f"{_polarity_word(z_score, 'avoids imperative openings', 'starts many sentences as commands')}"
    return f"{feature_name.replace('_', ' ')} ({value:.2f})"


def _summary_clause(feature_ranks: list[tuple[str, float, float]]) -> str:
    tags: list[str] = []
    for feature_name, value, z_score in feature_ranks:
        tag = _summary_tag(feature_name, value, z_score)
        if tag not in tags:
            tags.append(tag)
        if len(tags) == 3:
            break
    return ", ".join(tags[:2]) + (f", and {tags[2]}" if len(tags) > 2 else "")


def _summary_tag(feature_name: str, value: float, z_score: float) -> str:
    del value
    if feature_name == "sentence_length_mean":
        return "compact" if z_score < 0 else "long-form"
    if feature_name == "sentence_length_std":
        return "steady" if z_score < 0 else "rhythmic"
    if feature_name == "comma_rate_per_1000":
        return "comma-light" if z_score < 0 else "comma-rich"
    if feature_name == "semicolon_rate_per_1000":
        return "semicolon-light" if z_score < 0 else "semicolon-rich"
    if feature_name == "emdash_rate_per_1000":
        return "dash-light" if z_score < 0 else "dashy"
    if feature_name == "exclamation_rate_per_1000":
        return "restrained" if z_score < 0 else "emphatic"
    if feature_name == "question_rate_per_1000":
        return "declarative" if z_score < 0 else "questioning"
    if feature_name == "paren_rate_per_1000":
        return "clean" if z_score < 0 else "aside-rich"
    if feature_name == "contraction_rate_per_100_words":
        return "formal" if z_score < 0 else "conversational"
    if feature_name == "type_token_ratio":
        return "repetitive" if z_score < 0 else "varied"
    if feature_name == "mean_word_length":
        return "plain-spoken" if z_score < 0 else "polysyllabic"
    if feature_name == "uppercase_word_rate_per_100_words":
        return "calm" if z_score < 0 else "shouty"
    if feature_name == "digit_rate_per_1000":
        return "textual" if z_score < 0 else "numeric"
    if feature_name == "newline_rate_per_1000":
        return "flowing" if z_score < 0 else "fragmented"
    if feature_name == "paragraph_rate_per_1000":
        return "continuous" if z_score < 0 else "sectioned"
    if feature_name.startswith("func_"):
        word = feature_name.removeprefix("func_").removesuffix("_per_100_words")
        return f"{word}-heavy"
    if feature_name == "readability_flesch_ease":
        return "dense" if z_score < 0 else "easy-to-read"
    if feature_name == "readability_fk_grade":
        return "accessible" if z_score < 0 else "advanced"
    if feature_name == "register_first_person_rate":
        return "impersonal" if z_score < 0 else "personal"
    if feature_name == "register_second_person_rate":
        return "detached" if z_score < 0 else "reader-facing"
    if feature_name == "register_contraction_rate":
        return "formal" if z_score < 0 else "conversational"
    if feature_name == "register_passive_proxy":
        return "active" if z_score < 0 else "passive"
    if feature_name == "register_hedging_rate":
        return "assertive" if z_score < 0 else "hedged"
    if feature_name == "tone_exclamation_rate":
        return "restrained" if z_score < 0 else "emphatic"
    if feature_name == "tone_question_rate":
        return "declarative" if z_score < 0 else "questioning"
    if feature_name == "tone_intensifier_rate":
        return "measured" if z_score < 0 else "intense"
    if feature_name == "tone_parenthetical_rate":
        return "tight" if z_score < 0 else "aside-heavy"
    if feature_name == "discourse_transition_rate":
        return "plain" if z_score < 0 else "well-connected"
    if feature_name == "discourse_example_rate":
        return "abstract" if z_score < 0 else "example-rich"
    if feature_name == "discourse_contrast_rate":
        return "non-contrastive" if z_score < 0 else "contrastive"
    if feature_name == "discourse_opener_conjunction_rate":
        return "sentence-led" if z_score < 0 else "conjunction-led"
    if feature_name == "discourse_bullet_rate":
        return "paragraph-led" if z_score < 0 else "list-driven"
    if feature_name == "cognitive_concrete_example_density":
        return "abstract" if z_score < 0 else "concrete"
    if feature_name == "cognitive_sentence_len_cv":
        return "steady" if z_score < 0 else "rhythmic"
    if feature_name == "cognitive_imperative_rate":
        return "descriptive" if z_score < 0 else "directive"
    return feature_name.replace("_", "-")


def _polarity_word(z_score: float, low: str, high: str) -> str:
    return high if z_score >= 0 else low


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
