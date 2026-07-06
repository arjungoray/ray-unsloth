"""Run small evals against any SamplingClient-compatible object."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ray_unsloth.clients._remote import resolve
from ray_unsloth.evals.scorers import get_scorer
from ray_unsloth.types import ModelInput, SamplingParams


@dataclass(slots=True)
class EvalItem:
    prompt: str | list[int]
    expected: str | float | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EvalItem":
        prompt = data.get("prompt", data.get("input", data.get("question", "")))
        expected = data.get("expected", data.get("answer"))
        metadata = {key: value for key, value in data.items() if key not in {"prompt", "input", "question", "expected", "answer"}}
        return cls(prompt=prompt, expected=expected, metadata=metadata)

    def to_payload(self) -> dict[str, Any]:
        payload = {"prompt": self.prompt, **self.metadata}
        if self.expected is not None:
            payload["expected"] = self.expected
        return payload


@dataclass(slots=True)
class EvalSpec:
    name: str
    dataset: str | list[dict[str, Any]] | list[EvalItem]
    scorer: str = "contains"
    max_samples: int | None = None
    sampling_params: SamplingParams = field(default_factory=lambda: SamplingParams(max_tokens=64, temperature=0.0))
    checkpoint_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Match training-time tokenization: most SFT paths encode prompts with
    # special tokens (BOS), and evaluating without them shifts completions.
    add_special_tokens: bool = True


@dataclass(slots=True)
class EvalReport:
    id: str
    name: str
    scorer: str
    score: float
    rows: list[dict[str, Any]]
    created_at: float
    checkpoint_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RegressionGate:
    min_score: float | None = None
    max_drop: float | None = None

    def check(self, report: EvalReport, *, baseline: EvalReport | None = None) -> tuple[bool, list[str]]:
        failures: list[str] = []
        if self.min_score is not None and report.score < self.min_score:
            failures.append(f"score {report.score:.4f} is below min_score {self.min_score:.4f}")
        if self.max_drop is not None and baseline is not None:
            drop = baseline.score - report.score
            if drop > self.max_drop:
                failures.append(f"score drop {drop:.4f} exceeds max_drop {self.max_drop:.4f}")
        return not failures, failures


def load_dataset(dataset: str | list[dict[str, Any]] | list[EvalItem], *, max_samples: int | None = None) -> list[EvalItem]:
    if isinstance(dataset, str):
        path = Path(dataset).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Eval dataset does not exist: {path}")
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            loaded = json.loads(path.read_text())
            rows = loaded if isinstance(loaded, list) else loaded.get("items", [])
        items = [EvalItem.from_mapping(row) for row in rows]
    else:
        items = [item if isinstance(item, EvalItem) else EvalItem.from_mapping(item) for item in dataset]
    return items[:max_samples] if max_samples is not None else items


def _prompt_to_input(prompt: str | list[int], tokenizer: Any | None, *, add_special_tokens: bool = True) -> ModelInput:
    if isinstance(prompt, list):
        return ModelInput.from_ints(prompt)
    if tokenizer is not None:
        encoded = tokenizer(prompt, add_special_tokens=add_special_tokens)
        return ModelInput.from_ints(encoded["input_ids"])
    return ModelInput.from_ints(prompt.encode("utf-8", errors="replace"))


def run_eval(sampling_client: Any, spec: EvalSpec, *, store: Any | None = None, run_id: str | None = None) -> EvalReport:
    scorer = get_scorer(spec.scorer)
    tokenizer = None
    get_tokenizer = getattr(sampling_client, "get_tokenizer", None)
    if callable(get_tokenizer):
        tokenizer = resolve(get_tokenizer())
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(load_dataset(spec.dataset, max_samples=spec.max_samples)):
        sample = resolve(
            sampling_client.sample(
                _prompt_to_input(item.prompt, tokenizer, add_special_tokens=spec.add_special_tokens),
                num_samples=1,
                sampling_params=spec.sampling_params,
            )
        )
        text = sample.sequences[0].text or ""
        payload = item.to_payload()
        score = float(scorer(text, payload))
        rows.append({"index": index, "prompt": item.prompt, "expected": item.expected, "output": text, "score": score})
    aggregate = sum(row["score"] for row in rows) / len(rows) if rows else 0.0
    report = EvalReport(
        id=f"eval-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        name=spec.name,
        scorer=spec.scorer,
        score=aggregate,
        rows=rows,
        created_at=time.time(),
        checkpoint_path=spec.checkpoint_path,
        metadata={"run_id": run_id, **spec.metadata} if run_id else dict(spec.metadata),
    )
    if store is not None:
        store.record_eval(report.to_dict())
    return report


def compare_reports(candidate: EvalReport, baseline: EvalReport) -> dict[str, Any]:
    return {
        "candidate": candidate.id,
        "baseline": baseline.id,
        "candidate_score": candidate.score,
        "baseline_score": baseline.score,
        "delta": candidate.score - baseline.score,
    }
