import importlib.util
import py_compile
from pathlib import Path

import yaml

from ray_unsloth import TensorData


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "tinker_first_sft_training.py"
SPEC = importlib.util.spec_from_file_location("tinker_first_sft_training", EXAMPLE_PATH)
assert SPEC is not None
tinker_first_sft_training = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tinker_first_sft_training)


class MappingTemplateTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        del messages, tokenize
        if add_generation_prompt:
            return {"input_ids": [[10, 11]]}
        return {"input_ids": [[10, 11, 12, 99]]}

    def __call__(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return {"input_ids": [[13, 14]]}

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return ",".join(str(token) for token in tokens)


class LongPromptTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        del messages, tokenize
        prompt = list(range(100))
        if add_generation_prompt:
            return prompt
        return prompt + [200, 201]

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


def read_example() -> str:
    return EXAMPLE_PATH.read_text(encoding="utf-8")


def test_example_is_valid_python():
    py_compile.compile(str(EXAMPLE_PATH), doraise=True)


def test_example_imports_only_this_repo_api_not_tinker_packages():
    source = read_example()

    assert "import tinker" not in source
    assert "tinker_cookbook" not in source
    assert "from ray_unsloth import" in source
    assert "ServiceClient(config=args.config)" in source
    assert "AdamParams(learning_rate=learning_rate)" in source
    assert "SamplingParams(" in source


def test_conversation_to_datum_uses_shifted_target_tokens_and_weights():
    datum = tinker_first_sft_training.conversation_to_datum(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        MappingTemplateTokenizer(),
        max_length=16,
    )

    assert datum.model_input.to_ints() == [10, 11, 12]
    assert isinstance(datum.loss_fn_inputs["target_tokens"], TensorData)
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [11, 12, 99]
    assert datum.loss_fn_inputs["weights"].tolist() == [0.0, 1.0, 1.0]


def test_conversation_to_datum_preserves_assistant_tokens_when_truncating():
    datum = tinker_first_sft_training.conversation_to_datum(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        LongPromptTokenizer(),
        max_length=16,
    )

    assert datum.model_input.to_ints() == list(range(86, 99)) + [200]
    assert datum.loss_fn_inputs["target_tokens"].tolist() == list(range(87, 99)) + [200, 201]
    assert datum.loss_fn_inputs["weights"].tolist() == [0.0] * 12 + [1.0, 1.0]


def test_load_local_settings_reads_optional_yaml_settings(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "examples": {
                    "tinker_first_sft_training": {
                        "steps": 3,
                        "learning_rate": 0.5,
                        "max_length": 32,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    settings = tinker_first_sft_training.load_local_settings(config_path)

    assert settings["steps"] == 3
    assert settings["learning_rate"] == 0.5
    assert settings["max_length"] == 32


def test_weighted_mean_nll_ignores_padded_logprobs():
    datum = tinker_first_sft_training.conversation_to_datum(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        MappingTemplateTokenizer(),
        max_length=16,
    )
    loss = tinker_first_sft_training.weighted_mean_nll(
        [datum],
        [
            {
                "logprobs": TensorData(
                    data=[0.0, -2.0, -4.0, -1000.0, -1000.0],
                    dtype="float32",
                    shape=[5],
                )
            }
        ],
    )

    assert loss == 3.0


def test_example_uses_direct_async_sampling_result():
    source = read_example()

    assert "result = await sampling_client.sample_async(" in source
    assert "sample_future = await sampling_client.sample_async(" not in source
    assert "await sample_future.result_async()" not in source
