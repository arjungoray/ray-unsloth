import json
from pathlib import Path

import pytest
import yaml

from ray_unsloth.cli import main as cli_main
from ray_unsloth.config import RuntimeConfig


def _fake_config(tmp_path: Path) -> dict[str, object]:
    return {
        "provider": "fake",
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "model": {"base_model": "Test/Test-1B", "max_seq_length": 128},
        "lora": {"rank": 4, "random_state": 123},
        "tracking": True,
    }


def test_coded_error_renders_code_and_docs_anchor():
    with pytest.raises(ValueError) as exc_info:
        RuntimeConfig.from_dict({"speed": {"profile": "bogus"}})

    message = str(exc_info.value)
    assert "RU-1001" in message
    assert "Hint:" in message
    assert "Docs: https://arjungoray.github.io/ray-unsloth/reference/errors#ru-1001" in message


def test_errors_catalog_command_exits_zero(capsys):
    assert cli_main(["errors"]) == 0
    out = capsys.readouterr().out
    assert "RU-1001" in out
    assert "DESCRIPTION" in out


def test_schema_command_emits_json(capsys):
    assert cli_main(["schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert "provider" in schema["properties"]
    assert "model" in schema["properties"]
    assert "lora" in schema["properties"]
    assert "base_model" in schema["properties"]["model"]["properties"]
    assert "rank" in schema["properties"]["lora"]["properties"]


def test_init_writes_schema_and_config_loads(tmp_path: Path):
    config_path = tmp_path / "ray-unsloth.yaml"

    assert cli_main(["init", str(config_path), "--force"]) == 0

    schema_path = tmp_path / "ray-unsloth.schema.json"
    assert config_path.exists()
    assert schema_path.exists()
    assert (
        config_path.read_text(encoding="utf-8").splitlines()[0]
        == "# yaml-language-server: $schema=./ray-unsloth.schema.json"
    )

    config = RuntimeConfig.from_file(config_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert config.provider == "fake"
    assert "provider" in schema["properties"]
    assert "model" in schema["properties"]
    assert "lora" in schema["properties"]
    assert "base_model" in schema["properties"]["model"]["properties"]
    assert "rank" in schema["properties"]["lora"]["properties"]


def test_doctor_json_returns_parseable_rows(tmp_path: Path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_fake_config(tmp_path)), encoding="utf-8")

    assert cli_main(["--config", str(config_path), "doctor", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert isinstance(rows, list)
    assert rows
    assert all({"name", "status", "detail", "fix"} <= row.keys() for row in rows)


def test_up_fake_provider_outputs_checkpoint_path(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert cli_main(["up", "--provider", "fake", "--yes", "--steps", "3"]) == 0
    out = capsys.readouterr().out
    assert "Checkpoint:" in out
    assert "checkpoints" in out
    assert "Serve UI:" in out
