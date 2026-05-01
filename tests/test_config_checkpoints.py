from pathlib import Path

from ray_unsloth.checkpoints import (
    atomic_checkpoint_dir,
    base_manifest,
    checkpoint_ref,
    read_manifest,
    resolve_path,
    write_manifest,
)
from ray_unsloth.clients.rest import RestClient
from ray_unsloth.config import RuntimeConfig


def test_runtime_config_from_dict():
    config = RuntimeConfig.from_dict(
        {
            "model": {"base_model": "test/model"},
            "lora": {"rank": 8},
            "resources": {"sampler_replicas": 2},
            "modal": {"enabled": True, "gpu": "T4", "volume_name": "test-volume"},
        }
    )

    assert config.model.base_model == "test/model"
    assert config.lora.rank == 8
    assert config.resources.sampler_replicas == 2
    assert config.modal.enabled is True
    assert config.modal.gpu == "T4"
    assert config.modal.volume_name == "test-volume"


def test_runtime_config_resolves_model_specific_configs():
    config = RuntimeConfig.from_dict(
        {
            "model": {"base_model": "default/model", "max_seq_length": 1024},
            "lora": {"rank": 32, "target_modules": ["default_proj"]},
            "model_configs": {
                "qwen3.5-4b": {
                    "model": {
                        "base_model": "Qwen/Qwen3.5-4B",
                        "max_seq_length": 4096,
                        "load_in_4bit": False,
                    },
                    "lora": {
                        "rank": 16,
                        "target_modules": ["q_proj", "k_proj", "v_proj"],
                    },
                }
            },
        }
    )

    model, lora = config.resolve_model_configs("qwen3.5-4b")
    same_model, same_lora = config.resolve_model_configs("Qwen/Qwen3.5-4B")
    unknown_model, unknown_lora = config.resolve_model_configs("other/model")

    assert model.base_model == "Qwen/Qwen3.5-4B"
    assert model.max_seq_length == 4096
    assert model.load_in_4bit is False
    assert lora.rank == 16
    assert lora.target_modules == ["q_proj", "k_proj", "v_proj"]
    assert same_model == model
    assert same_lora == lora
    assert unknown_model.base_model == "other/model"
    assert unknown_model.max_seq_length == 1024
    assert unknown_lora.rank == 32


def test_runtime_config_uses_selected_model_config_as_default():
    config = RuntimeConfig.from_dict(
        {
            "model": {"config": "qwen3.5-4b"},
            "lora": {"rank": 32, "target_modules": ["default_proj"]},
            "model_configs": {
                "qwen3.5-4b": {
                    "model": {
                        "base_model": "Qwen/Qwen3.5-4B",
                        "max_seq_length": 2048,
                        "load_in_4bit": False,
                    },
                    "lora": {
                        "rank": 16,
                        "target_modules": ["q_proj", "k_proj", "v_proj"],
                    },
                }
            },
        }
    )

    model, lora = config.resolve_model_configs()

    assert config.default_model_config == "qwen3.5-4b"
    assert model.base_model == "Qwen/Qwen3.5-4B"
    assert model.max_seq_length == 2048
    assert model.load_in_4bit is False
    assert lora.rank == 16
    assert lora.target_modules == ["q_proj", "k_proj", "v_proj"]


def test_atomic_checkpoint_manifest(tmp_path: Path):
    target = tmp_path / "checkpoint"

    with atomic_checkpoint_dir(target) as workdir:
        (workdir / "weights.txt").write_text("ok", encoding="utf-8")
        write_manifest(
            workdir,
            base_manifest(
                kind="training_state",
                step=3,
                base_model="base",
                lora={"rank": 4},
                has_optimizer=True,
            ),
        )

    manifest = read_manifest(target)
    ref = checkpoint_ref(target, has_optimizer=True)

    assert (target / "weights.txt").read_text(encoding="utf-8") == "ok"
    assert manifest["step"] == 3
    assert ref.has_optimizer is True


def test_tinker_local_path_resolution(tmp_path: Path):
    assert resolve_path(f"tinker://local/{tmp_path}") == tmp_path.resolve()


def test_rest_client_lists_and_publishes_checkpoints(tmp_path: Path):
    target = tmp_path / "state-step-1"
    target.mkdir()
    write_manifest(
        target,
        base_manifest(
            kind="training_state",
            step=1,
            base_model="base",
            lora={"rank": 4},
            has_optimizer=False,
            extra={"session_id": "train-1", "metadata": {"owner": "test"}},
        ),
    )
    rest = RestClient(config=RuntimeConfig(checkpoint_root=str(tmp_path)))

    checkpoints = rest.list_checkpoints("train-1").result().checkpoints
    run = rest.get_training_run("train-1").result()
    published = rest.publish_checkpoint_from_tinker_path(f"tinker://local/{target}").result()

    assert checkpoints[0].path == str(target)
    assert run.metadata == {"owner": "test"}
    assert published.metadata["published"] is True
