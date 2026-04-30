import importlib.util
from pathlib import Path

import pytest


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "overfit_smoke_test.py"
SPEC = importlib.util.spec_from_file_location("overfit_smoke_test", EXAMPLE_PATH)
assert SPEC is not None
overfit_smoke_test = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(overfit_smoke_test)

build_sft_datum = overfit_smoke_test.build_sft_datum
assert_meaningful_generation = overfit_smoke_test.assert_meaningful_generation


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=True):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


def test_build_sft_datum_masks_prompt_tokens():
    datum, prompt_input, target_token_count = build_sft_datum(FakeTokenizer(), "Question:", " answer")

    assert prompt_input.to_ints() == [ord(char) for char in "Question:"]
    assert datum.model_input.to_ints() == [ord(char) for char in "Question: answer"]
    assert datum.loss_fn_inputs["labels"] == [-100] * len("Question:") + [
        ord(char) for char in " answer"
    ]
    assert target_token_count == len(" answer")


def test_assert_meaningful_generation_rejects_punctuation_only():
    with pytest.raises(AssertionError, match="only punctuation"):
        assert_meaningful_generation("!!!!!!", " blue maple.")


def test_assert_meaningful_generation_requires_expected_answer():
    with pytest.raises(AssertionError, match="expected canary"):
        assert_meaningful_generation("This is a fluent but wrong answer.", " blue maple.")


def test_assert_meaningful_generation_accepts_canary_answer():
    assert_meaningful_generation(
        "The ray-unsloth smoke test answer is blue maple.",
        " blue maple.",
    )
