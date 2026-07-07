"""Tests for Scribe's style-transfer (rewrite) core."""

import json

import pytest
import yaml

from ray_unsloth.cli import main as cli_main
from ray_unsloth.store import RunStore
from ray_unsloth_apps.scribe.rewrite import (
    RewritePair,
    content_jaccard,
    length_ratio_score,
    load_pairs,
    rewrite_prompt,
    rule_based_neutralize,
    save_pairs,
)

VOICE_PASSAGES = [
    "ok so here's the thing... i tried the new build and tbh it's fine. not great. fine.",
    "ngl the meeting could've been an email. again. we keep doing this and i keep saying it...",
    "quick note... shipped the fix. tests pass. if it breaks blame the cache, it's always the cache.",
    "tbh i don't hate the new design. the spacing is off tho. like, everywhere. we should fix that.",
    "so the demo went ok... mostly. the wifi died mid-slide, classic. we recovered. barely.",
    "reminder... standup is at ten. bring actual updates this time. 'still working on it' is not an update.",
    "i read the doc. it's long. tbh it could be a third of the size and say more.",
    "the bug is back. of course it's back. it never left. it was just... waiting.",
    "ngl the offsite was fun. the trust falls were not. my back agrees.",
    "coffee machine is broken again... this is week three. i am not ok about it.",
    "shipped v2 last night. quiet launch. no fires yet. i said YET.",
    "tbh the roadmap looks ambitious. which is a nice word for impossible. we'll see.",
]


def test_content_jaccard_tracks_meaning_not_style():
    original = "shipped the fix last night. tests pass. blame the cache if it breaks."
    neutral = "The fix was shipped last night and the tests pass; the cache may be responsible for any breakage."
    unrelated = "The quarterly budget review covered marketing spend and hiring plans."
    assert content_jaccard(original, neutral) > 0.3
    assert content_jaccard(original, unrelated) < 0.1


def test_length_ratio_score_window():
    source = "one two three four five six seven eight nine ten"
    assert length_ratio_score(source, source) == 1.0
    assert length_ratio_score(source, "one two") < 1.0
    assert length_ratio_score(source, " ".join(["word"] * 60)) == 0.0


def test_rule_based_neutralize_strips_style_keeps_content():
    original = "tbh the roadmap looks ambitious — which is a nice word for impossible... we'll see!"
    neutral = rule_based_neutralize(original)
    assert "tbh" not in neutral.lower()
    assert "—" not in neutral and "..." not in neutral and "!" not in neutral
    assert content_jaccard(original, neutral) > 0.5


def test_rewrite_prompt_is_stable_and_contains_source():
    prompt = rewrite_prompt("Some pasted text.")
    assert "Some pasted text." in prompt
    assert prompt == rewrite_prompt("Some pasted text.")
    assert prompt.endswith("Rewritten:\n")


def test_pairs_roundtrip(tmp_path):
    pairs = [RewritePair(source="plain text", target="styled text...", method="rule")]
    save_pairs(tmp_path / "pairs.jsonl", pairs)
    loaded = load_pairs(tmp_path / "pairs.jsonl")
    assert loaded == pairs


@pytest.fixture()
def rewrite_config(tmp_path):
    config = {
        "provider": "fake",
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "tracking": True,
        "model": {"base_model": "Test/Test-1B", "max_seq_length": 128},
        "lora": {"rank": 4, "random_state": 123},
        "scribe": {
            "sft_epochs": 1,
            "sft_batch_size": 4,
            "rl_rounds": 1,
            "group_size": 4,
            "batches_per_round": 2,
            "max_tokens": 32,
            "max_prompts": 6,
            "eval_generations": 10,
            "run_name": "scribe-rw",
            "seed": 7,
            "sft_learning_rate": 0.02,
            "rl_learning_rate": 1e-5,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, tmp_path


def test_rewrite_pipeline_end_to_end_on_fake_provider(rewrite_config, capsys):
    config_path, tmp_path = rewrite_config
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for index, passage in enumerate(VOICE_PASSAGES):
        (corpus / f"note_{index:02d}.txt").write_text(passage + "\n")

    assert (
        cli_main(
            [
                "--config",
                str(config_path),
                "scribe",
                "pipeline",
                str(corpus),
                "--workdir",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )
    summary_out = capsys.readouterr().out
    summary = json.loads(summary_out[summary_out.index("{") :])
    assert summary["pairs"]["n_pairs"] > 0

    pairs = load_pairs(tmp_path / "work" / "pairs.jsonl")
    assert pairs, "neutralization must produce rewrite pairs (rule fallback on fake)"
    assert all(pair.target for pair in pairs)
    assert any(pair.method in ("rule", "model") for pair in pairs)

    store = RunStore(tmp_path / "checkpoints")
    evals = store.list_evals()
    assert evals, "eval stage must record a report"
    latest = evals[0]
    assert latest.get("task") == "rewrite"
    assert "content_mean" in latest and 0.0 <= latest["content_mean"] <= 1.0

    # the rewrite CLI runs against the trained checkpoint
    assert (
        cli_main(
            [
                "--config",
                str(config_path),
                "scribe",
                "rewrite",
                "--text",
                "The deployment finished and the metrics look stable.",
                "--max-tokens",
                "24",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert output.strip(), "rewrite must print a candidate"
