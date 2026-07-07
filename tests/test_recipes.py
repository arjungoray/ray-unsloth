from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ray_unsloth import ServiceClient
from ray_unsloth.recipes import (
    GrpoConfig,
    PromptSpec,
    Renderer,
    Rollout,
    Rubric,
    RubricTerm,
    TrainOnWhat,
    drop_uniform_groups,
    group_relative,
    grpo_round,
    rollout_to_datum,
)


class _ChatTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        del tokenize
        if add_generation_prompt:
            return {"input_ids": [[10, 11]]}
        if len(messages) == 1:
            return {"input_ids": [[10, 11, 12]]}
        return {"input_ids": [[10, 11, 12, 13]]}

    def __call__(self, text, add_special_tokens=True):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


class _CompletionTokenizer:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=True):
        tokens = [ord(char) for char in text]
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return {"input_ids": tokens}


def _fake_config(tmp_path):
    return {
        "provider": "fake",
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "tracking": False,
        "model": {"base_model": "Test/Test-1B", "max_seq_length": 128},
        "lora": {"rank": 4, "random_state": 123},
    }


def test_renderer_masks_prompt_positions_and_trains_assistant_tokens():
    renderer = Renderer(name="chat_template")
    datum = renderer.build_sft_datum(
        _ChatTokenizer(),
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
        TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    assert datum.model_input.to_ints() == [10, 11, 12, 13]
    assert datum.loss_fn_inputs["labels"].tolist()[:2] == [-100, -100]
    assert datum.loss_fn_inputs["weights"].tolist()[:2] == [0.0, 0.0]
    assert datum.loss_fn_inputs["weights"].tolist()[2:] == [1.0, 1.0]


def test_rollout_to_datum_matches_policy_contract():
    datum = rollout_to_datum(
        [101, 102, 103],
        Rollout(tokens=[201, 202], text="ok", logprobs=[-0.4, -0.5]),
        advantage=0.75,
    )

    assert datum.model_input.to_ints() == [101, 102, 103, 201]
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [-100, -100, 201, 202]
    assert datum.loss_fn_inputs["logprobs"].tolist() == pytest.approx([0.0, 0.0, -0.4, -0.5])
    assert datum.loss_fn_inputs["advantages"].tolist() == pytest.approx([0.0, 0.0, 0.75, 0.75])
    assert datum.loss_fn_inputs["weights"].tolist() == pytest.approx([0.0, 0.0, 0.5, 0.5])


def test_rubric_z_normalization_and_override():
    rubric = Rubric(
        terms=[
            RubricTerm(
                name="score",
                fn=lambda *, score, **_: float(score),
                weight=1.0,
                z_normalize=True,
            ),
            RubricTerm(
                name="override",
                fn=lambda *, flag, **_: float(flag),
                weight=1.0,
                z_normalize=False,
                override_below=0.0,
            ),
        ]
    )

    breakdowns = rubric.score(
        [
            {"completion_text": "a", "completion_tokens": [1], "context": {"score": 1.0, "flag": 1.0}},
            {"completion_text": "b", "completion_tokens": [2], "context": {"score": 3.0, "flag": -1.0}},
        ]
    )

    assert breakdowns[0].terms["score"] == 0.0
    assert breakdowns[1].terms["score"] == pytest.approx(1.0)
    assert breakdowns[1].total == -1.0


def test_group_relative_math():
    assert group_relative([1.0, 0.0, 1.0]) == pytest.approx([1 / 3, -2 / 3, 1 / 3])
    assert group_relative([1.0, 1.0], normalize_std=True) == [0.0, 0.0]
    assert drop_uniform_groups([[1.0, 1.0], [1.0, 1.2], []], threshold=0.05) == [[1.0, 1.2]]


def test_grpo_round_on_fake_provider_has_datums_and_finite_losses(tmp_path):
    service = ServiceClient(config=_fake_config(tmp_path))
    training = service.create_lora_training_client()
    prompt_bank = [
        PromptSpec(prompt_text="Reply with OK.", context={"target": "OK"}),
        PromptSpec(prompt_text="Say OK.", context={"target": "OK"}),
        PromptSpec(prompt_text="Return OK.", context={"target": "OK"}),
    ]
    rubric = Rubric(
        terms=[
            RubricTerm(
                name="parity",
                fn=lambda *, completion_tokens, **_: float(sum(completion_tokens) % 2),
                weight=1.0,
                z_normalize=False,
            ),
            RubricTerm(
                name="brevity",
                fn=lambda *, completion_text, **_: -len(completion_text) / 1000.0,
                weight=1.0,
                z_normalize=False,
            ),
        ]
    )

    report = grpo_round(
        training,
        prompt_bank,
        rubric,
        GrpoConfig(
            group_size=3, prompts_per_batch=2, batches_per_round=2, inner_epochs=1, learning_rate=0.02, max_tokens=6
        ),
    )
    service.close()

    assert report.n_datums > 0
    assert report.losses and all(math.isfinite(loss) for loss in report.losses)


def test_gallery_examples_run_via_subprocess_smoke(tmp_path):
    # Subprocess smoke: keep the supported examples runnable end to end.
    env = dict(os.environ, PYTHONPATH=str(Path.cwd() / "src"))
    root = Path(__file__).resolve().parents[1] / "examples" / "gallery"
    for script in (root / "01-hello-sft" / "run.py", root / "02-hello-rl" / "run.py"):
        result = subprocess.run(
            [sys.executable, str(script), "--smoke"],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
