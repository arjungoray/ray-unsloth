from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from ray_unsloth.cli import main as cli_main
from ray_unsloth.recipes.renderers import Renderer
from ray_unsloth.store import RunStore
from ray_unsloth_apps.scribe.classifier import auc as classifier_auc
from ray_unsloth_apps.scribe.classifier import train_classifier
from ray_unsloth_apps.scribe.ingest import Passage, ingest_paths, scrub
from ray_unsloth_apps.scribe.pipeline import run_pipeline
from ray_unsloth_apps.scribe.profile import build_profile, copy_overlap, stylometrics, stylometry_distance
from ray_unsloth_apps.scribe.prompts import backtranslate_prompts


def _scribe_config(tmp_path: Path, **scribe_overrides):
    scribe = {
        "sft_epochs": 1,
        "sft_batch_size": 4,
        "rl_rounds": 1,
        "group_size": 4,
        "batches_per_round": 2,
        "max_tokens": 32,
        "max_prompts": 8,
        "eval_generations": 40,
        "run_name": "scribe",
        "seed": 7,
    }
    scribe.update(scribe_overrides)
    return {
        "provider": "fake",
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "tracking": True,
        "model": {"base_model": "Test/Test-1B", "max_seq_length": 128},
        "lora": {"rank": 4, "random_state": 123},
        "scribe": scribe,
    }


def _approx_tokens(text: str) -> int:
    words = re.findall(r"\b[\w']+\b", text)
    return max(1, round(len(words) * 1.3))


def test_ingest_recursively_merges_and_dedupes_and_scrub(tmp_path: Path):
    root = tmp_path / "corpus"
    txt = root / "alpha"
    md = root / "beta"
    txt.mkdir(parents=True)
    md.mkdir(parents=True)
    (txt / "one.txt").write_text(
        "first small passage with enough words for the window.\n\n"
        "second small passage with enough words for the window.\n\n"
        "first small passage with enough words for the window.\n",
        encoding="utf-8",
    )
    (md / "two.md").write_text(
        "third small passage with enough words for the window.\n\n"
        "fourth small passage with enough words for the window.\n",
        encoding="utf-8",
    )

    passages = ingest_paths([str(root)], min_tokens=5, max_tokens=20)

    assert len(passages) == len({passage.text for passage in passages})
    assert all(5 <= _approx_tokens(passage.text) <= 20 for passage in passages)
    assert scrub("email me at person@example.com or visit https://example.com") == "email me at <EMAIL> or visit <URL>"


def test_profile_stylometrics_deterministic_and_distance_orders_texts():
    passages = [
        Passage(
            text="i keep it short and lowercase, tbh, with tiny sentences and a soft cadence...", source="a", kind="txt"
        ),
        Passage(
            text="still small, still lowercase, still a little elliptical, and still pretty calm...",
            source="b",
            kind="txt",
        ),
        Passage(
            text="another brief note, all casual and low-key, tbh, with a relaxed rhythm...", source="c", kind="txt"
        ),
    ]
    profile = build_profile(passages)
    text = passages[0].text
    shuffled = " ".join(reversed(text.split()))

    assert stylometrics(text) == stylometrics(text)
    assert stylometry_distance(text, profile) < stylometry_distance(shuffled, profile)
    assert copy_overlap(text, profile) == pytest.approx(1.0, abs=1e-6)


def test_classifier_separates_synthetic_styles():
    pos = [
        "short sentence. stop. okay.",
        "tiny line. quick note. done.",
        "brief note. clear message. end.",
        "small sentence. casual tone. tbh.",
    ] * 8
    neg = [
        "the quarterly report, which was reviewed carefully, offers a broader, calmer explanation of the same point.",
        "in the longer memo, the writer, taking a measured approach, continues with commas, clauses, and detail.",
        "the summary, although concise in spirit, unfolds through long, careful phrasing, with pauses and elaboration.",
        "this paragraph, with its deliberate rhythm and nested clauses, stays extended and comma-heavy throughout.",
    ] * 8

    model = train_classifier(pos, neg, epochs=200, lr=0.1, seed=3)

    assert classifier_auc(pos, neg) > 0.9
    assert model.predict_proba(pos[0]) > model.predict_proba(neg[0])


@dataclass
class _Immediate:
    value: object

    def result(self):
        return self.value


class _ToyTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [byte for byte in text.encode("utf-8", errors="replace")]}


@dataclass
class _ToySequence:
    text: str


@dataclass
class _ToyResponse:
    sequences: list[_ToySequence]


class _ToySamplingClient:
    def __init__(self, text: str):
        self.text = text

    def get_tokenizer(self):
        return _Immediate(_ToyTokenizer())

    def sample(self, *args, **kwargs):
        del args, kwargs
        return _Immediate(_ToyResponse([_ToySequence(self.text)]))


def test_prompts_fallback_and_leak_filter():
    renderer = Renderer(name="plain")
    passages = [
        Passage(text="short text one.", source="a", kind="txt"),
        Passage(text="short text two.", source="b", kind="txt"),
    ]
    fallback_client = _ToySamplingClient("   ")
    prompts = backtranslate_prompts(fallback_client, passages, renderer, max_prompts=8)
    assert len(prompts) == len(passages)
    assert all(row["source"] == "backtranslated" for row in prompts)

    leaky_passage = Passage(
        text="alpha beta gamma delta epsilon zeta eta theta iota",
        source="c",
        kind="txt",
    )
    leaky_client = _ToySamplingClient(leaky_passage.text)
    assert backtranslate_prompts(leaky_client, [leaky_passage], renderer, max_prompts=8) == []


def test_pipeline_end_to_end_fake_provider(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for index in range(12):
        text = (
            f"tbh this little note {index} stays lowercase and breezy ... "
            "it keeps the tone relaxed, a bit elliptical, and pretty direct."
        )
        suffix = ".md" if index % 2 else ".txt"
        (corpus / f"passage-{index}{suffix}").write_text(text, encoding="utf-8")

    config_path = tmp_path / "scribe.yaml"
    config_path.write_text(yaml.safe_dump(_scribe_config(tmp_path)), encoding="utf-8")

    summary = run_pipeline(str(config_path), tmp_path / "scribe", steps=None, paths=[str(corpus)])

    store = RunStore(tmp_path / "checkpoints")
    assert summary["sft"]["before_loss"] > summary["sft"]["after_loss"]
    assert summary["rl"]["rounds"][0]["n_datums"] > 0
    assert math.isfinite(summary["eval"]["auc"])
    assert store.list_runs(app="scribe")
    assert Path(summary["export"]["output_path"], "Modelfile").exists()


def test_cli_apps_and_pipeline_command(tmp_path: Path, capsys):
    corpus = tmp_path / "cli-corpus"
    corpus.mkdir()
    for index in range(4):
        (corpus / f"item-{index}.txt").write_text(
            f"tbh the cli passage {index} keeps it short, casual, and a little elliptical...",
            encoding="utf-8",
        )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_scribe_config(tmp_path)), encoding="utf-8")

    assert cli_main(["apps", "--json"]) == 0
    apps = json.loads(capsys.readouterr().out)
    assert any(row["name"] == "scribe" for row in apps)

    assert (
        cli_main(
            [
                "--config",
                str(config_path),
                "scribe",
                "pipeline",
                str(corpus),
                "--workdir",
                str(tmp_path / "workdir"),
            ]
        )
        == 0
    )
