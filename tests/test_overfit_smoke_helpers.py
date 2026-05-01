import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from ray_unsloth import GeneratedSequence, ModelInput, SampleResponse
from ray_unsloth.runtime.unsloth.engine import UnslothEngine


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "overfit_smoke_test.py"
SPEC = importlib.util.spec_from_file_location("overfit_smoke_test", EXAMPLE_PATH)
assert SPEC is not None
overfit_smoke_test = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(overfit_smoke_test)

build_sft_datum = overfit_smoke_test.build_sft_datum
assert_meaningful_generation = overfit_smoke_test.assert_meaningful_generation
assert_sampling_features = overfit_smoke_test.assert_sampling_features


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=True):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


class FakeTokenizerWithEos:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=True):
        tokens = [ord(char) for char in text]
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return {"input_ids": tokens}


class FakeProcessorWithoutEncode:
    def __call__(self, text=None, add_special_tokens=True):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


class FakeNestedTokenizer:
    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return [ord(char) for char in text]


class FakeProcessorWithNestedTokenizer:
    tokenizer = FakeNestedTokenizer()


def test_build_sft_datum_masks_prompt_tokens():
    datum, prompt_input, target_token_count = build_sft_datum(FakeTokenizer(), "Question:", " answer")

    assert prompt_input.to_ints() == [ord(char) for char in "Question:"]
    assert datum.model_input.to_ints() == [ord(char) for char in "Question: answer"]
    assert datum.loss_fn_inputs["labels"] == [-100] * len("Question:") + [
        ord(char) for char in " answer"
    ]
    assert target_token_count == len(" answer")


def test_build_sft_datum_strips_prompt_only_eos():
    datum, prompt_input, target_token_count = build_sft_datum(FakeTokenizerWithEos(), "Question:", " answer")

    assert prompt_input.to_ints() == [ord(char) for char in "Question:"]
    assert datum.model_input.to_ints() == [ord(char) for char in "Question: answer"] + [99]
    assert datum.loss_fn_inputs["labels"] == [-100] * len("Question:") + [
        ord(char) for char in " answer"
    ] + [99]
    assert target_token_count == len(" answer") + 1


def test_stop_token_ids_supports_processors_without_encode():
    engine = object.__new__(UnslothEngine)
    engine.tokenizer = FakeProcessorWithoutEncode()

    assert UnslothEngine._stop_token_ids(engine, ["."]) == [[ord(".")]]


def test_stop_token_ids_prefers_nested_text_tokenizer():
    engine = object.__new__(UnslothEngine)
    engine.tokenizer = FakeProcessorWithNestedTokenizer()

    assert UnslothEngine._stop_token_ids(engine, ["."]) == [[ord(".")]]


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


def test_assert_sampling_features_accepts_new_sampling_fields():
    prompt = ModelInput.from_ints([1, 2, 3])
    response = SampleResponse(
        sequences=[
            GeneratedSequence(
                tokens=[4, 5],
                text="ok",
                logprobs=[-0.1, -0.2],
                stop_reason="length",
            )
        ],
        prompt_logprobs=[None, -0.4, -0.5],
        topk_prompt_logprobs=[[], [(2, -0.4)], [(3, -0.5)]],
    )

    assert_sampling_features(response, prompt, max_tokens=2)


def test_assert_sampling_features_rejects_missing_prompt_logprobs():
    prompt = ModelInput.from_ints([1])
    response = SampleResponse(
        sequences=[GeneratedSequence(tokens=[2], logprobs=[-0.1], stop_reason="length")],
        topk_prompt_logprobs=[[]],
    )

    with pytest.raises(AssertionError, match="prompt_logprobs"):
        assert_sampling_features(response, prompt, max_tokens=1)


def test_sample_with_feature_checks_requests_new_sampling_features():
    calls = []

    class FakeSampler:
        def sample(
            self,
            prompt_input,
            num_samples=1,
            sampling_params=None,
            include_prompt_logprobs=False,
            topk_prompt_logprobs=0,
        ):
            calls.append(
                {
                    "prompt_input": prompt_input,
                    "num_samples": num_samples,
                    "sampling_params": sampling_params,
                    "include_prompt_logprobs": include_prompt_logprobs,
                    "topk_prompt_logprobs": topk_prompt_logprobs,
                }
            )
            return SimpleNamespace(
                result=lambda: SampleResponse(
                    sequences=[
                        GeneratedSequence(
                            tokens=[4],
                            text="ok",
                            logprobs=[-0.1],
                            stop_reason="stop",
                        )
                    ],
                    prompt_logprobs=[None, -0.2],
                    topk_prompt_logprobs=[[], [(4, -0.2)]],
                )
            )

    prompt = ModelInput.from_ints([1, 2])
    response = overfit_smoke_test.sample_with_feature_checks(FakeSampler(), prompt, max_tokens=3)

    assert response.sequences[0].tokens == [4]
    assert calls[0]["include_prompt_logprobs"] is True
    assert calls[0]["topk_prompt_logprobs"] == 3
    assert calls[0]["sampling_params"].stop == ["."]
