import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from ray_unsloth.config import RuntimeConfig


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "lfm2_5_1_2b_multitenant_rl.py"
CONFIG_PATH = Path(__file__).parents[1] / "configs" / "lfm2_5_1_2b_1x_l4_multitenant_rl.yaml"
SPEC = importlib.util.spec_from_file_location("lfm2_5_1_2b_multitenant_rl", EXAMPLE_PATH)
lfm2_5_1_2b_multitenant_rl = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lfm2_5_1_2b_multitenant_rl
SPEC.loader.exec_module(lfm2_5_1_2b_multitenant_rl)


def test_multitenant_rl_config_uses_one_shared_l4_trainer_pool():
    config = RuntimeConfig.from_file(CONFIG_PATH)
    settings = lfm2_5_1_2b_multitenant_rl.load_local_settings(CONFIG_PATH)

    assert config.modal.enabled is True
    assert config.modal.gpu == "L4"
    assert config.modal.max_inputs == 3
    assert config.modal.trainer_pool_key == "lfm2.5-1.2b-instruct-l4-shared"
    assert config.resources.trainer_replicas == 3
    assert config.default_model_config == "lfm2.5-1.2b-instruct"
    assert config.model.attn_implementation == "flash_attention_2"
    assert 25 <= settings["steps"] <= 30
    assert settings["wandb"]["enabled"] is True
    assert settings["wandb"]["project"] == "ray-unsloth-multitenant"


def test_multitenant_rl_example_runs_tenants_with_importance_sampling():
    source = EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "asyncio.gather" in source
    assert "create_lora_training_client_async" in source
    assert "collect_rollouts" in source
    assert 'loss_fn="importance_sampling"' in source
    assert "save_weights_for_sampler_async" in source
    assert 'reinit="create_new"' in source
    assert '"aggregate/tokens/train_total"' in source
    assert '"aggregate/tokens/sample_total"' in source
    assert '"aggregate/tokens/prefill_total"' in source


def test_multitenant_rl_tenant_specs_are_independent_same_model_runs():
    specs = lfm2_5_1_2b_multitenant_rl.tenant_specs()

    assert len(specs) == 3
    assert {spec.name for spec in specs} == {
        "algebra-rl-tenant",
        "arith-rl-tenant",
        "word-problem-rl-tenant",
    }
    assert len({spec.seed for spec in specs}) == 3
    assert all(len(spec.problems) >= 4 for spec in specs)


def test_multitenant_rl_wandb_logger_logs_event_index_and_train_step():
    class FakeRun:
        def __init__(self):
            self.calls = []

        def log(self, payload, step):
            self.calls.append((payload, step))

    run = FakeRun()
    logger = lfm2_5_1_2b_multitenant_rl.WandbRunLogger(enabled=True, run=run)

    logger.log({"reward/mean": 0.5}, step=2)

    assert run.calls == [({"wandb/event_index": 1, "train/step": 2, "reward/mean": 0.5}, 1)]


def test_multitenant_rl_token_accounting_helpers():
    rollouts = [
        SimpleNamespace(
            prompt=SimpleNamespace(length=11),
            completions=[
                SimpleNamespace(tokens=[1, 2, 3]),
                SimpleNamespace(tokens=[4]),
            ],
        ),
        SimpleNamespace(
            prompt=SimpleNamespace(length=7),
            completions=[
                SimpleNamespace(tokens=[5, 6]),
            ],
        ),
    ]
    datums = [
        SimpleNamespace(model_input=SimpleNamespace(length=13)),
        SimpleNamespace(model_input=SimpleNamespace(length=17)),
    ]

    assert lfm2_5_1_2b_multitenant_rl.rollout_prefill_token_count(rollouts) == 18
    assert lfm2_5_1_2b_multitenant_rl.rollout_token_count(rollouts) == 6
    assert lfm2_5_1_2b_multitenant_rl.train_token_count(datums) == 30
