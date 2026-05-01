import importlib.util
import py_compile
import sys
from pathlib import Path

import pytest
import yaml

from ray_unsloth import ModelInput, TensorData


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "tinker_first_rl_training.py"
SPEC = importlib.util.spec_from_file_location("tinker_first_rl_training", EXAMPLE_PATH)
assert SPEC is not None
tinker_first_rl_training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tinker_first_rl_training
assert SPEC.loader is not None
SPEC.loader.exec_module(tinker_first_rl_training)


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
    assert 'loss_fn="importance_sampling"' in source
    assert "SamplingParams(" in source


def test_load_local_settings_reads_optional_yaml_settings(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "examples": {
                    "tinker_first_rl_training": {
                        "steps": 2,
                        "batch_size": 3,
                        "group_size": 4,
                        "learning_rate": 0.5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    settings = tinker_first_rl_training.load_local_settings(config_path)

    assert settings["steps"] == 2
    assert settings["batch_size"] == 3
    assert settings["group_size"] == 4
    assert settings["learning_rate"] == 0.5


def test_grade_answer_uses_last_boxed_numeric_answer():
    assert tinker_first_rl_training.extract_boxed("first \\boxed{2}; final \\boxed{1,234}") == "1,234"
    assert tinker_first_rl_training.grade_answer("so the answer is \\boxed{1,234}", "1234") == 1.0
    assert tinker_first_rl_training.grade_answer("so the answer is 1234", "1234") == 0.0
    assert tinker_first_rl_training.grade_answer("so the answer is \\boxed{1235}", "1234") == 0.0


def test_group_relative_advantages_center_rewards():
    advantages = tinker_first_rl_training.group_relative_advantages([1.0, 0.0, 1.0])

    assert advantages == pytest.approx([1 / 3, -2 / 3, 1 / 3])
    assert sum(advantages) == pytest.approx(0.0)


def test_build_policy_datum_aligns_prompt_completion_targets_and_advantages():
    datum = tinker_first_rl_training.build_policy_datum(
        prompt=ModelInput.from_ints([101, 102, 103]),
        completion_tokens=[201, 202],
        completion_logprobs=[-0.4, -0.5],
        advantage=0.75,
    )

    assert datum.model_input.to_ints() == [101, 102, 103, 201]
    assert isinstance(datum.loss_fn_inputs["target_tokens"], TensorData)
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [0, 0, 201, 202]
    assert datum.loss_fn_inputs["logprobs"].tolist() == pytest.approx([0.0, 0.0, -0.4, -0.5])
    assert datum.loss_fn_inputs["advantages"].tolist() == pytest.approx([0.0, 0.0, 0.75, 0.75])
    assert datum.loss_fn_inputs["weights"].tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])


def test_build_policy_datum_pads_missing_generated_logprobs():
    datum = tinker_first_rl_training.build_policy_datum(
        prompt=ModelInput.from_ints([101, 102]),
        completion_tokens=[201, 202, 203],
        completion_logprobs=[None, -0.5],
        advantage=-0.25,
    )

    assert datum.model_input.to_ints() == [101, 102, 201, 202]
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [0, 201, 202, 203]
    assert datum.loss_fn_inputs["logprobs"].tolist() == pytest.approx([0.0, 0.0, -0.5, 0.0])
    assert datum.loss_fn_inputs["advantages"].tolist() == pytest.approx([0.0, -0.25, -0.25, -0.25])


def test_rollout_to_datums_skips_degenerate_groups():
    problem = tinker_first_rl_training.MathProblem("What is 1 + 1?", "2")
    degenerate_rollout = tinker_first_rl_training.ProblemRollout(
        problem=problem,
        prompt=ModelInput.from_ints([101, 102]),
        completions=[
            tinker_first_rl_training.GradedCompletion(
                tokens=[201],
                logprobs=[-0.2],
                text="\\boxed{2}",
                reward=1.0,
                advantage=0.0,
            )
        ],
        mean_reward=1.0,
        degenerate=True,
    )
    useful_rollout = tinker_first_rl_training.ProblemRollout(
        problem=problem,
        prompt=ModelInput.from_ints([101, 102]),
        completions=[
            tinker_first_rl_training.GradedCompletion(
                tokens=[201],
                logprobs=[-0.2],
                text="\\boxed{2}",
                reward=1.0,
                advantage=0.5,
            ),
            tinker_first_rl_training.GradedCompletion(
                tokens=[202],
                logprobs=[-0.3],
                text="\\boxed{3}",
                reward=0.0,
                advantage=-0.5,
            ),
        ],
        mean_reward=0.5,
        degenerate=False,
    )

    assert tinker_first_rl_training.rollout_to_datums(degenerate_rollout) == []
    assert len(tinker_first_rl_training.rollout_to_datums(useful_rollout)) == 2


def test_policy_loss_summary_uses_nonzero_policy_outputs():
    mean_logprob, mean_ratio = tinker_first_rl_training.policy_loss_summary(
        [
            {
                "logprobs": TensorData(data=[0.0, -1.0, -3.0], dtype="float32", shape=[3]),
                "ratios": TensorData(data=[0.0, 0.5, 1.5], dtype="float32", shape=[3]),
            }
        ]
    )

    assert mean_logprob == pytest.approx(-2.0)
    assert mean_ratio == pytest.approx(1.0)


def test_example_uses_direct_async_sampling_result():
    source = read_example()

    assert "sample_results = await asyncio.gather(*sample_coros)" in source
    assert "sample_future = await sampling_client.sample_async(" not in source
    assert "await sample_future.result_async()" not in source
