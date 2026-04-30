from pathlib import Path

from ray_unsloth.checkpoints import (
    atomic_checkpoint_dir,
    base_manifest,
    checkpoint_ref,
    read_manifest,
    write_manifest,
)
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
