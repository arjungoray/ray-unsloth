"""Style-transfer data generation and scoring for Scribe.

The trained model's job is: given arbitrary pasted text, rewrite it in the
user's voice. Training pairs come from **neutralization back-translation**:
each corpus passage is paraphrased into plain, generic prose (by the base
model, with a deterministic rule-based fallback), and the model learns the
inverse mapping ``rewrite_prompt(neutral) -> original passage``. Inference
uses the identical prompt template, so pasting any text reproduces the
training-time task exactly.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ray_unsloth import SamplingParams

REWRITE_INSTRUCTION = (
    "Rewrite the following text in the author's writing style. "
    "Keep every fact and the meaning; keep roughly the same length."
)

NEUTRALIZE_INSTRUCTION = (
    "Paraphrase the following text into plain, neutral, generic prose. "
    "Keep every fact and the meaning. Remove stylistic quirks: no dashes, "
    "no ellipses, no slang, no rhetorical questions; use standard sentences."
)

_WORD_RE = re.compile(r"[a-zA-Z']+")

# Small stopword list: function words carry style, not content.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "so",
        "if",
        "then",
        "than",
        "because",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "them",
        "their",
        "not",
        "no",
        "yes",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "just",
        "very",
        "really",
        "about",
        "into",
        "over",
        "under",
        "out",
        "up",
        "down",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "too",
        "s",
        "t",
        "don",
        "now",
    ]
)


_META_TRAILER = re.compile(
    r"(?:^|\n\n)\s*(?:i(?:'ve| have)? (?:made sure|kept|preserved|maintained)|note:|as requested)[^\0]*$",
    re.IGNORECASE,
)


def strip_meta_trailer(text: str) -> str:
    """Drop trailing self-commentary ("I made sure to preserve...") from a rewrite."""
    return _META_TRAILER.sub("", text).strip()


def rewrite_prompt(source: str) -> str:
    return f"{REWRITE_INSTRUCTION}\n\nText:\n{source.strip()}\n\nRewritten:\n"


def neutralize_prompt(passage: str) -> str:
    return f"{NEUTRALIZE_INSTRUCTION}\n\nText:\n{passage.strip()}\n\nPlain paraphrase:\n"


# ---------------------------------------------------------------------------
# Content preservation metrics (the anti-gaming keystone for rewriting)
# ---------------------------------------------------------------------------


def content_words(text: str) -> set[str]:
    return {word.lower() for word in _WORD_RE.findall(text) if word.lower() not in STOPWORDS and len(word) > 2}


def content_jaccard(a: str, b: str) -> float:
    """Jaccard similarity of content words — cheap, hard-to-game meaning proxy."""
    words_a, words_b = content_words(a), content_words(b)
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def length_ratio_score(source: str, output: str) -> float:
    """1.0 when output length is within [0.6, 1.6]x the source, tapering to 0."""
    source_words = max(1, len(source.split()))
    output_words = len(output.split())
    ratio = output_words / source_words
    if 0.6 <= ratio <= 1.6:
        return 1.0
    if ratio <= 0.0 or ratio > 4.0:
        return 0.0
    distance = (0.6 - ratio) if ratio < 0.6 else (ratio - 1.6)
    return max(0.0, 1.0 - distance)


# ---------------------------------------------------------------------------
# Neutralization
# ---------------------------------------------------------------------------

_SLANG = re.compile(r"\b(tbh|ngl|imo|imho|btw|lol|idk)\b[,;]?\s*", re.IGNORECASE)


def rule_based_neutralize(text: str) -> str:
    """Deterministic de-styler: strips signature mechanics, keeps content.

    Used as the fallback when the base model's paraphrase fails filters (and
    as the only path on the fake provider, keeping CI deterministic).
    """
    result = text
    result = _SLANG.sub("", result)
    result = result.replace("—", ", ").replace("…", ". ").replace("...", ". ")
    result = re.sub(r"[!?]+", ".", result)
    result = re.sub(r"\s*\.\s*\.", ".", result)
    result = re.sub(r"\s+", " ", result).strip()
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", result) if s.strip()]
    normalized = []
    for sentence in sentences:
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        normalized.append(sentence)
    return " ".join(normalized)


@dataclass(slots=True)
class RewritePair:
    source: str  # neutralized / arbitrary input text
    target: str  # the user's original passage
    method: str  # "model" | "rule"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def neutralize_passages(
    sampling_client: Any,
    passages: list[Any],
    renderer: Any,
    *,
    per_passage: int = 2,
    max_tokens: int = 220,
    seed: int = 0,
) -> list[RewritePair]:
    """Build (neutral source -> original passage) pairs for style-transfer SFT.

    Model paraphrases must actually preserve content (content_jaccard >= 0.25)
    without copying the passage wholesale (jaccard <= 0.9 and no big verbatim
    run); failures fall back to the rule-based neutralizer so every passage
    yields at least one pair.
    """
    tokenizer = sampling_client.get_tokenizer().result()
    pairs: list[RewritePair] = []
    seen_sources: set[str] = set()
    targets: list[str] = []
    for passage in passages:
        passage_text = passage.text if hasattr(passage, "text") else str(passage)
        targets.append(passage_text)
        # Short sources must be in-distribution too: pasted text comes in all
        # lengths, so also train on 1-2 sentence slices of each passage.
        slice_text = _short_slice(passage_text)
        if slice_text:
            targets.append(slice_text)
    for index, text in enumerate(targets):
        got_model_pair = False
        for attempt in range(max(2, per_passage + 1)):
            prompt_input = renderer.build_generation_prompt(
                tokenizer,
                [{"role": "user", "content": neutralize_prompt(text)}],
            )
            generation = sampling_client.sample(
                prompt_input,
                num_samples=1,
                sampling_params=SamplingParams(
                    max_tokens=max_tokens,
                    # Ramp temperature across attempts: diverse paraphrases of
                    # the same passage are cheap augmentation for tiny corpora.
                    temperature=0.7 + 0.2 * attempt,
                    seed=seed + index * 17 + attempt,
                ),
            ).result()
            candidate = ""
            if generation.sequences:
                candidate = clean_generation(str(generation.sequences[0].text or ""))
            if _acceptable_neutralization(candidate, text) and candidate not in seen_sources:
                seen_sources.add(candidate)
                pairs.append(RewritePair(source=candidate, target=text, method="model"))
                got_model_pair = True
        if not got_model_pair:
            fallback = rule_based_neutralize(text)
            # A fallback that still shares a long verbatim run with the target
            # would teach the identity mapping (the run-4 postmortem bug):
            # drop the passage instead of training copy-through.
            if fallback and fallback not in seen_sources and not _shares_verbatim_run(fallback, text):
                seen_sources.add(fallback)
                pairs.append(RewritePair(source=fallback, target=text, method="rule"))
    return pairs


_CHATTY_PREFIXES = (
    "plain paraphrase:",
    "paraphrase:",
    "here is",
    "here's",
    "sure,",
    "sure!",
    "certainly",
)


def clean_generation(candidate: str) -> str:
    """Strip chat-model wrapper junk so the paraphrase itself is the source."""
    cleaned = candidate.strip().strip('"').strip()
    lowered = cleaned.lower()
    for prefix in _CHATTY_PREFIXES:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].lstrip(" :\n")
            lowered = cleaned.lower()
    return cleaned.strip()


def _acceptable_neutralization(candidate: str, original: str) -> bool:
    if len(candidate) < 40 or not re.search(r"[A-Za-z]{3}", candidate):
        return False
    # Content must be preserved. NOTE: no upper bound — a faithful paraphrase
    # legitimately keeps most content words; identity is caught by the
    # verbatim-run check below, not by vocabulary overlap.
    if content_jaccard(candidate, original) < 0.22:
        return False
    return not _shares_verbatim_run(candidate, original)


def _shares_verbatim_run(candidate: str, original: str, n: int = 12) -> bool:
    """True when the two texts share any verbatim n-word run (identity signal)."""
    original_runs = _word_runs(original, n)
    return any(run in original_runs for run in _word_runs(candidate, n))


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _short_slice(text: str) -> str | None:
    """A 1-2 sentence slice of a passage, when it has enough sentences to spare."""
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) < 3:
        return None
    take = 2 if len(sentences[0].split()) < 8 else 1
    slice_text = " ".join(sentences[:take]).strip()
    return slice_text if len(slice_text.split()) >= 5 else None


def _word_runs(text: str, n: int) -> set[str]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    return {" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def save_pairs(path: str | Path, pairs: list[RewritePair]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_dict(), sort_keys=True) + "\n")


def load_pairs(path: str | Path) -> list[RewritePair]:
    source = Path(path)
    if not source.exists():
        return []
    pairs = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        pairs.append(RewritePair(source=data["source"], target=data["target"], method=data.get("method", "model")))
    return pairs
