"""Run Qwen3.5 4B GRPO-style math RL on one Modal L4.

This is the same math-dataset training example as
``qwen3_5_9b_math_dataset_rl_training.py``, configured for the smaller
Qwen3.5 4B model and a single L4. It also logs cumulative W&B token counters:

* ``tokens/prefill_total``: prompt tokens sent to sampling calls
* ``tokens/sample_total``: generated completion tokens returned by sampling
* ``tokens/train_total``: tokens submitted to training forward/backward calls

Run:

    python examples/qwen3_5_4b_math_dataset_rl_training.py \
        --config configs/qwen3_5_4b_1x_l4.yaml \
        --dataset math \
        --dataset-limit 256
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml


_RECIPE_PATH = Path(__file__).with_name("qwen3_5_9b_math_dataset_rl_training.py")
_RECIPE_SPEC = importlib.util.spec_from_file_location("qwen3_5_9b_math_dataset_rl_training_recipe", _RECIPE_PATH)
if _RECIPE_SPEC is None or _RECIPE_SPEC.loader is None:
    raise ImportError(f"Could not load math dataset recipe from {_RECIPE_PATH}")
recipe = importlib.util.module_from_spec(_RECIPE_SPEC)
sys.modules[_RECIPE_SPEC.name] = recipe
_RECIPE_SPEC.loader.exec_module(recipe)


BASE_MODEL = "qwen3.5-4b"
SAMPLER_NAME = "qwen3.5-4b-math-dataset-rl"
WANDB_RUN_NAME = "qwen3.5-4b-1xl4-math-dataset-rl"
SETTINGS_KEY = "qwen3_5_4b_math_dataset_rl_training"


def load_local_settings(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return dict(data.get("examples", {}).get(SETTINGS_KEY, {}))


async def train(args: argparse.Namespace) -> None:
    recipe.BASE_MODEL = BASE_MODEL
    recipe.SAMPLER_NAME = SAMPLER_NAME
    recipe.WANDB_RUN_NAME = WANDB_RUN_NAME
    recipe.load_local_settings = load_local_settings
    await recipe.train(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_5_4b_1x_l4.yaml")
    parser.add_argument("--dataset", choices=["math", "gsm8k", "deepmath", "polaris"], default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--dataset-limit", dest="dataset_limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--group-size", dest="group_size", type=int, default=None)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=None)
    parser.add_argument("--train-microbatch-size", dest="train_microbatch_size", type=int, default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=None)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    args = parser.parse_args()

    asyncio.run(train(args))


if __name__ == "__main__":
    main()
