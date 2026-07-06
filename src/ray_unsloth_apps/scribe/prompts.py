"""Prompt generation helpers for Scribe."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ray_unsloth import SamplingParams
from ray_unsloth.recipes.renderers import Renderer
from ray_unsloth_apps.scribe.ingest import Passage

NEUTRAL_PARAGRAPHS = [
    "The quarterly report summarizes revenue, support volume, and product delivery across the previous month.",
    "The team maintained the documentation site, updated examples, and corrected a handful of outdated references.",
    "A standard operating review covered release timing, incident follow-up, and the remaining action items.",
    "The office notice described building access, visitor registration, and the schedule for maintenance work.",
    "Project planning included timeline estimates, a dependency map, and a short discussion of risks.",
    "The meeting notes recorded a budget review, an implementation update, and a reminder about next steps.",
    "The internal wiki page explains the data export process, the approval flow, and the support contact path.",
    "The operations summary mentions backup verification, permission checks, and routine housekeeping tasks.",
    "The policy memo describes document retention, approval thresholds, and the expected response window.",
    "The launch recap notes the rollout sequence, the feedback collected from users, and the follow-up plan.",
]

_WORD_RE = re.compile(r"\b[\w']+\b")


def backtranslate_prompts(
    sampling_client: Any,
    passages: list[Passage],
    renderer: Renderer,
    *,
    max_prompts: int,
) -> list[dict[str, Any]]:
    tokenizer = sampling_client.get_tokenizer().result()
    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, passage in enumerate(passages[:max_prompts]):
        template = (
            "Given this text, write the instruction a person would have given an assistant to produce it. "
            "Include form, approximate length, audience, and tone.\nText: {passage}"
        )
        prompt_input = renderer.build_generation_prompt(
            tokenizer,
            [{"role": "user", "content": template.format(passage=passage.text)}],
        )
        generation = sampling_client.sample(
            prompt_input,
            num_samples=1,
            sampling_params=SamplingParams(max_tokens=48, temperature=0.0, seed=index),
        ).result()
        text = ""
        if generation.sequences:
            text = str(generation.sequences[0].text or "")
        text = text.strip()
        if len(text) < 8 or not re.search(r"[A-Za-z]", text):
            text = _fallback_prompt(passage.text)
        if _shares_8gram(text, passage.text):
            continue
        if text in seen:
            continue
        seen.add(text)
        prompts.append(
            {
                "text": text,
                "source": "backtranslated",
                "passage_source": passage.source,
                "passage_text": passage.text,
            }
        )
    return prompts


def task_bank() -> list[str]:
    return [
        "Write a brief email declining a meeting about quarterly planning.",
        "Write a short note declining an invitation to review a budget draft.",
        "Write a concise paragraph declining a request to join an extra committee.",
        "Write a brief email apologizing for a late reply about a documentation review.",
        "Write a short note apologizing for missing a project update call.",
        "Write a concise paragraph apologizing for a delay in sending meeting notes.",
        "Write a brief email recapping the action items from a launch review.",
        "Write a short note recapping the decisions made during a status meeting.",
        "Write a concise paragraph recapping the main points from a team discussion.",
        "Write a brief email asking for a small schedule change next week.",
        "Write a short note asking for clarification on a policy reminder.",
        "Write a concise paragraph asking for feedback on a draft announcement.",
        "Write a brief email explaining a delay in a normal office process.",
        "Write a short note explaining why a document needs one more review.",
        "Write a concise paragraph explaining a change to the support workflow.",
        "Write a brief email announcing a routine maintenance window.",
        "Write a short note announcing a revised timeline for a deliverable.",
        "Write a concise paragraph announcing a new documentation update.",
        "Write a brief email declining a lunch invitation in a polite way.",
        "Write a short note apologizing for a missed check-in on a calendar issue.",
        "Write a concise paragraph recapping a product demo for a general audience.",
        "Write a brief email asking for a quick status update on a task.",
        "Write a short note explaining a small change to a travel plan.",
        "Write a concise paragraph announcing an internal wiki cleanup.",
        "Write a brief email declining a request to add one more meeting.",
        "Write a short note apologizing for confusion in a project summary.",
        "Write a concise paragraph asking for help with a shared spreadsheet.",
        "Write a brief email recapping the outcome of a vendor call.",
        "Write a short note explaining the purpose of a new reminder process.",
        "Write a concise paragraph announcing a change in office hours.",
    ]


def save_prompts(prompts: list[dict[str, Any]], path: str | Path) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt, sort_keys=True) + "\n")


def load_prompts(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(dict(json.loads(line)))
    return rows


def _fallback_prompt(passage_text: str) -> str:
    words = _WORD_RE.findall(passage_text)
    preview = " ".join(words[:8])
    word_count = max(1, round(len(words)))
    return f"Write a passage of about {word_count} words in the author's usual style about: {preview}..."


def _shares_8gram(left: str, right: str) -> bool:
    left_hashes = _shingle_hashes(left)
    return bool(left_hashes & _shingle_hashes(right))


def _shingle_hashes(text: str) -> set[int]:
    words = [word.lower() for word in _WORD_RE.findall(text)]
    if len(words) < 8:
        return set()
    hashes: set[int] = set()
    for start in range(0, len(words) - 8 + 1):
        shingle = " ".join(words[start : start + 8])
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        hashes.add(int.from_bytes(digest, "big"))
    return hashes
