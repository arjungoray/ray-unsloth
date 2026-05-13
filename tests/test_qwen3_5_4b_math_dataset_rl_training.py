import importlib.util
import py_compile
import sys
from pathlib import Path

from ray_unsloth.config import RuntimeConfig


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "qwen3_5_4b_math_dataset_rl_training.py"
CONFIG_PATH = Path(__file__).parents[1] / "configs" / "qwen3_5_4b_1x_l4.yaml"
SPEC = importlib.util.spec_from_file_location("qwen3_5_4b_math_dataset_rl_training", EXAMPLE_PATH)
assert SPEC is not None
qwen3_5_4b_math_dataset_rl_training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen3_5_4b_math_dataset_rl_training
assert SPEC.loader is not None
SPEC.loader.exec_module(qwen3_5_4b_math_dataset_rl_training)


def test_qwen3_5_4b_math_dataset_example_is_valid_python():
    py_compile.compile(str(EXAMPLE_PATH), doraise=True)


def test_qwen3_5_4b_config_uses_one_l4_and_4b_model():
    config = RuntimeConfig.from_file(CONFIG_PATH)
    settings = qwen3_5_4b_math_dataset_rl_training.load_local_settings(CONFIG_PATH)

    assert config.modal.enabled is True
    assert config.modal.gpu == "L4"
    assert config.default_model_config == "qwen3.5-4b"
    assert config.model.base_model == "Qwen/Qwen3.5-4B"
    assert config.model.max_seq_length == 2048
    assert config.model.load_in_4bit is False
    assert config.model.fast_inference == "auto"
    assert config.lora.rank == 16
    assert settings["batch_size"] == 1
    assert settings["group_size"] == 4
    assert settings["max_tokens"] == 8096
    assert settings["max_train_tokens"] == 2048
    assert settings["train_microbatch_size"] == 1
    assert settings["wandb"]["name"] == "qwen3.5-4b-1xl4-math-dataset-rl"


def test_qwen3_5_4b_wrapper_points_recipe_at_4b_settings(monkeypatch):
    calls = {}

    async def fake_train(args):
        calls["base_model"] = qwen3_5_4b_math_dataset_rl_training.recipe.BASE_MODEL
        calls["sampler_name"] = qwen3_5_4b_math_dataset_rl_training.recipe.SAMPLER_NAME
        calls["settings"] = qwen3_5_4b_math_dataset_rl_training.recipe.load_local_settings(args.config)

    monkeypatch.setattr(qwen3_5_4b_math_dataset_rl_training.recipe, "train", fake_train)

    import asyncio
    import argparse

    asyncio.run(qwen3_5_4b_math_dataset_rl_training.train(argparse.Namespace(config=CONFIG_PATH)))

    assert calls["base_model"] == "qwen3.5-4b"
    assert calls["sampler_name"] == "qwen3.5-4b-math-dataset-rl"
    assert calls["settings"]["name"] == "qwen3.5-4b-math-dataset-rl"
