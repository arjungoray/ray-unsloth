import importlib.util
import py_compile
import sys
from pathlib import Path
from types import SimpleNamespace

from ray_unsloth.config import RuntimeConfig


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "qwen3_5_4b_ruler_64k_rl_training.py"
CONFIG_PATH = Path(__file__).parents[1] / "configs" / "qwen3_5_4b_ruler_64k.yaml"
SPEC = importlib.util.spec_from_file_location("qwen3_5_4b_ruler_64k_rl_training", EXAMPLE_PATH)
assert SPEC is not None
qwen3_5_4b_ruler_64k_rl_training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen3_5_4b_ruler_64k_rl_training
assert SPEC.loader is not None
SPEC.loader.exec_module(qwen3_5_4b_ruler_64k_rl_training)


def test_qwen3_5_4b_ruler_64k_example_is_valid_python():
    py_compile.compile(str(EXAMPLE_PATH), doraise=True)


def test_qwen3_5_4b_ruler_64k_config_uses_same_4b_model_and_long_context():
    config = RuntimeConfig.from_file(CONFIG_PATH)
    settings = qwen3_5_4b_ruler_64k_rl_training.load_local_settings(CONFIG_PATH)

    assert config.modal.enabled is True
    assert config.modal.gpu == "A100-80GB"
    assert config.default_model_config == "qwen3.5-4b-ruler-64k"
    assert config.model.base_model == "Qwen/Qwen3.5-4B"
    assert config.model.max_seq_length == 81920
    assert config.model.load_in_4bit is False
    assert config.model.fast_inference == "auto"
    assert config.model.attn_implementation == "flash_attention_2"
    assert config.lora.rank == 8
    assert settings["dataset_name"] == "tonychenxyz/ruler-full"
    assert settings["dataset_config"] == "plain"
    assert settings["task"] == "niah_single_1"
    assert settings["context_length"] == 65536
    assert settings["min_prompt_tokens"] == 40000
    assert settings["max_tokens"] == 4096
    assert settings["dataset_scan_limit"] == 40000
    assert settings["max_train_tokens"] == 73728
    assert settings["train_microbatch_size"] == 1
    assert settings["degenerate_reward_baseline"] == 0.0


def test_ruler_reward_requires_all_answers():
    score_ruler_response = qwen3_5_4b_ruler_64k_rl_training.score_ruler_response

    assert score_ruler_response("The magic numbers are 123 and 456.", ["123", "456"]) == 1.0
    assert score_ruler_response("The magic number is 123.", ["123", "456"]) == 0.0


def test_load_ruler_problems_filters_64k_rows(monkeypatch):
    rows = [
        {
            "prompt": "short",
            "category": "plain/ruler/niah_single_1_4096",
            "extra_info": {
                "ground_truth": {"answers": ["111"]},
                "ruler_task": "niah_single_1",
                "context_length": 4096,
            },
        },
        {
            "prompt": "long",
            "category": "plain/ruler/niah_single_1_65536",
            "extra_info": {
                "ground_truth": {"answers": ["222"]},
                "ruler_task": "niah_single_1",
                "context_length": 65536,
            },
        },
    ]

    def load_dataset(*args, **kwargs):
        assert args[:3] == ("tonychenxyz/ruler-full", "plain")
        assert kwargs == {"split": "validation", "streaming": True}
        return iter(rows)

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [7] * (45000 if text == "long" else 100)

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))

    problems, info = qwen3_5_4b_ruler_64k_rl_training.load_ruler_problems(
        tokenizer=Tokenizer(),
        dataset_name="tonychenxyz/ruler-full",
        dataset_config="plain",
        split="validation",
        task="niah_single_1",
        context_length=65536,
        limit=1,
        scan_limit=8,
        min_prompt_tokens=40000,
        max_prompt_tokens=66000,
    )

    assert info.selected_rows == 1
    assert info.scanned_rows == 2
    assert problems[0].answers == ["222"]
    assert problems[0].prompt.length == 45000
    assert problems[0].category == "plain/ruler/niah_single_1_65536"
