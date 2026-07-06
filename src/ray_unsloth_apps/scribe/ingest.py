"""Corpus ingestion helpers for Scribe."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class Passage:
    text: str
    source: str
    kind: str


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+", re.MULTILINE)
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b[\w']+\b")


def scrub(text: str) -> str:
    text = _URL_RE.sub("<URL>", text)
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _PHONE_RE.sub("<PHONE>", text)
    return text


def ingest_paths(paths: list[str], *, min_tokens: int = 40, max_tokens: int = 400) -> list[Passage]:
    passages: list[Passage] = []
    seen: set[str] = set()
    for path in _expand_paths(paths):
        suffix = path.suffix.lower()
        kind = suffix.removeprefix(".") if suffix in {".txt", ".md"} else "text"
        raw = path.read_text(encoding="utf-8", errors="replace")
        paragraphs = [chunk for chunk in _PARA_SPLIT_RE.split(raw) if chunk.strip()]
        buffer: list[str] = []
        buffer_source = f"{path}:{0}"
        for index, paragraph in enumerate(paragraphs):
            cleaned = _clean_paragraph(paragraph)
            if not cleaned:
                continue
            if _approx_tokens(cleaned) > max_tokens:
                if buffer:
                    _emit_passage(passages, seen, " ".join(buffer), buffer_source, kind)
                    buffer = []
                _split_long_paragraph(passages, seen, cleaned, f"{path}:{index}", kind, max_tokens)
                continue
            tentative = " ".join([*buffer, cleaned]).strip()
            if buffer and _approx_tokens(tentative) > max_tokens:
                _emit_passage(passages, seen, " ".join(buffer), buffer_source, kind)
                buffer = [cleaned]
                buffer_source = f"{path}:{index}"
            else:
                if not buffer:
                    buffer_source = f"{path}:{index}"
                buffer.append(cleaned)
        if buffer:
            _emit_passage(passages, seen, " ".join(buffer), buffer_source, kind)
    return passages


def save_passages(passages: Iterable[Passage], path: str | Path) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for passage in passages:
            handle.write(json.dumps(asdict(passage), sort_keys=True) + "\n")


def load_passages(path: str | Path) -> list[Passage]:
    rows: list[Passage] = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(Passage(**json.loads(line)))
    return rows


def _expand_paths(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in {".txt", ".md"}:
                    files.append(candidate)
        elif path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            files.append(path)
    return sorted(dict.fromkeys(files))


def _clean_paragraph(text: str) -> str:
    return _WS_RE.sub(" ", text.strip())


def _approx_tokens(text: str) -> int:
    words = _WORD_RE.findall(text)
    return max(1, round(len(words) * 1.3))


def _text_key(text: str) -> str:
    normalised = _WS_RE.sub(" ", text).strip().lower()
    return hashlib.sha1(normalised.encode("utf-8", errors="replace")).hexdigest()


def _emit_passage(passages: list[Passage], seen: set[str], text: str, source: str, kind: str) -> None:
    cleaned = _clean_paragraph(text)
    if not cleaned:
        return
    key = _text_key(cleaned)
    if key in seen:
        return
    seen.add(key)
    passages.append(Passage(text=cleaned, source=source, kind=kind))


def _split_long_paragraph(
    passages: list[Passage],
    seen: set[str],
    text: str,
    source: str,
    kind: str,
    max_tokens: int,
) -> None:
    words = text.split()
    chunk_words = max(1, int(max_tokens / 1.3))
    for start in range(0, len(words), chunk_words):
        chunk = " ".join(words[start : start + chunk_words])
        if chunk:
            _emit_passage(passages, seen, chunk, f"{source}:{start // chunk_words}", kind)
