import importlib.util
import sys
from pathlib import Path

from ray_unsloth.config import RuntimeConfig


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "qwen3_5_4b_multitenant_sft.py"
CONFIG_PATH = Path(__file__).parents[1] / "configs" / "qwen3_5_4b_1x_a100_multitenant.yaml"
SPEC = importlib.util.spec_from_file_location("qwen3_5_4b_multitenant_sft", EXAMPLE_PATH)
qwen3_5_4b_multitenant_sft = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen3_5_4b_multitenant_sft
SPEC.loader.exec_module(qwen3_5_4b_multitenant_sft)


def test_multitenant_config_uses_one_shared_a100_trainer_pool():
    config = RuntimeConfig.from_file(CONFIG_PATH)
    settings = qwen3_5_4b_multitenant_sft.load_local_settings(CONFIG_PATH)

    assert config.modal.enabled is True
    assert config.modal.gpu == "A100"
    assert config.modal.max_inputs == 2
    assert config.modal.trainer_pool_key == "qwen3.5-4b-a100-shared"
    assert config.resources.trainer_replicas == 2
    assert config.default_model_config == "qwen3.5-4b"
    assert config.model.base_model == "Qwen/Qwen3.5-4B"
    assert settings["wandb"]["enabled"] is True
    assert settings["wandb"]["project"] == "ray-unsloth-multitenant"


def test_multitenant_example_runs_tenants_concurrently_and_reuses_live_actor_for_sampling():
    source = EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "asyncio.gather" in source
    assert "create_lora_training_client_async" in source
    assert "SamplingClient(" in source
    assert "actors=[training_client._actor]" in source
    assert 'reinit="create_new"' in source


def test_wandb_logger_logs_event_index_and_train_step():
    class FakeRun:
        def __init__(self):
            self.calls = []

        def log(self, payload, step):
            self.calls.append((payload, step))

    run = FakeRun()
    logger = qwen3_5_4b_multitenant_sft.WandbRunLogger(enabled=True, run=run)

    logger.log({"train/loss": 1.25}, step=3)

    assert run.calls == [({"wandb/event_index": 1, "train/step": 3, "train/loss": 1.25}, 1)]


def test_tenant_specs_are_independent_same_model_runs():
    specs = qwen3_5_4b_multitenant_sft.tenant_specs()

    assert len(specs) == 2
    assert {spec.name for spec in specs} == {"arc-math-tutor", "vale-writing-coach"}
    assert len({spec.seed for spec in specs}) == 2
