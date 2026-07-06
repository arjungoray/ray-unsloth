import asyncio
import json
from pathlib import Path

import pytest
import yaml

from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient
from ray_unsloth.cli import main as cli_main
from ray_unsloth.config import RuntimeConfig, load_config
from ray_unsloth.evals import EvalSpec, run_eval
from ray_unsloth.export import export_checkpoint, list_exporters
from ray_unsloth.providers import get_provider, list_providers, resolve_provider_name
from ray_unsloth.store import RunStore


def _config(tmp_path, **overrides):
    data = {
        "provider": "fake",
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "model": {"base_model": "Test/Test-1B", "max_seq_length": 128},
        "lora": {"rank": 4, "random_state": 123},
        "tracking": True,
    }
    data.update(overrides)
    return data


def _datum(text="abababab"):
    tokens = list(text.encode())
    return Datum(model_input=ModelInput.from_ints(tokens), loss_fn_inputs={"labels": tokens})


def test_fake_provider_training_records_metrics_and_checkpoint_lineage(tmp_path):
    service = ServiceClient(config=_config(tmp_path))
    trainer = service.create_lora_training_client()

    first = trainer.forward_backward([_datum()]).result().loss
    trainer.optim_step(AdamParams(learning_rate=0.4)).result()
    second = trainer.forward_backward([_datum()]).result().loss
    trainer.optim_step(AdamParams(learning_rate=0.4)).result()
    checkpoint = trainer.save_state("state-a").result()
    sampler_checkpoint = trainer.save_weights_for_sampler("sampler-a").result()
    service.close()

    assert second < first
    assert trainer.run_id is not None
    store = RunStore(tmp_path / "checkpoints")
    run = store.get_run(trainer.run_id)
    assert run is not None
    assert run.status == "completed"
    metrics = store.read_metrics(trainer.run_id)
    assert [row["event"] for row in metrics].count("forward_backward") == 2
    checkpoints = store.list_checkpoints()
    assert {record.kind for record in checkpoints} == {"training_state", "sampler"}
    lineage = store.lineage(checkpoint.path)
    assert [record.path for record in lineage] == [checkpoint.path]
    assert sampler_checkpoint.path


def test_store_schema_versions_round_trip_and_load_old_records(tmp_path):
    store = RunStore(tmp_path / "checkpoints")

    old_run_path = store._run_file("run-old")
    old_run_path.parent.mkdir(parents=True, exist_ok=True)
    old_run_path.write_text(
        json.dumps(
            {
                "id": "run-old",
                "name": "legacy",
                "status": "running",
                "provider": "fake",
                "base_model": "Test/Test-1B",
            },
            indent=2,
            sort_keys=True,
        )
    )

    checkpoints_path = store._checkpoints_file()
    checkpoints_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoints_path.write_text(
        json.dumps(
            {
                "path": "/tmp/legacy",
                "run_id": "run-old",
                "step": 1,
                "kind": "training_state",
            },
            sort_keys=True,
        )
        + "\n"
    )

    old_run = store.get_run("run-old")
    old_checkpoint = store.list_checkpoints()[0]
    new_run = store.create_run(provider="fake", base_model="Test/Test-1B")
    new_checkpoint = store.record_checkpoint(path="/tmp/new", run_id=new_run.id, step=2, kind="training_state")

    assert old_run is not None
    assert old_run.schema == 1
    assert old_checkpoint.schema == 1
    assert json.loads(store._run_file(new_run.id).read_text())["schema"] == 1
    assert json.loads(store._checkpoints_file().read_text().splitlines()[-1])["schema"] == 1
    assert new_checkpoint.schema == 1


def test_async_training_recording_preserves_two_stage_future_shape(tmp_path):
    async def run():
        service = ServiceClient(config=_config(tmp_path))
        trainer = await service.create_lora_training_client_async()
        submitted = await trainer.forward_backward_async([_datum()])
        output = await submitted.result_async()
        await (await trainer.optim_step_async(AdamParams(learning_rate=0.2))).result_async()
        service.close()
        return trainer.run_id, output.loss

    run_id, loss = asyncio.run(run())

    assert loss > 0
    assert RunStore(tmp_path / "checkpoints").read_metrics(run_id)


def test_training_run_context_manager_marks_completed(tmp_path):
    service = ServiceClient(config=_config(tmp_path))
    with service.training_run() as trainer:
        trainer.forward_backward([_datum()]).result()
        run_id = trainer.run_id
    service.close()

    assert run_id is not None
    run = RunStore(tmp_path / "checkpoints").get_run(run_id)
    assert run is not None
    assert run.status == "completed"


def test_training_run_context_manager_marks_failed(tmp_path):
    service = ServiceClient(config=_config(tmp_path))
    with pytest.raises(RuntimeError, match="boom"), service.training_run() as trainer:
        run_id = trainer.run_id
        raise RuntimeError("boom")
    service.close()

    assert run_id is not None
    run = RunStore(tmp_path / "checkpoints").get_run(run_id)
    assert run is not None
    assert run.status == "failed"


def test_provider_registry_validation_and_plans(tmp_path):
    config = RuntimeConfig.from_dict(_config(tmp_path, provider="skypilot", provider_options={"gpu": "L4"}))

    assert "fake" in list_providers()
    assert resolve_provider_name(config) == "skypilot"
    plan = get_provider("skypilot").plan(config)
    assert "task.yaml" in plan.artifacts
    assert plan.fit is not None
    assert config.validate() == []


def test_eval_and_export_workflows(tmp_path):
    service = ServiceClient(config=_config(tmp_path))
    trainer = service.create_lora_training_client()
    trainer.forward_backward([_datum("hello hello")]).result()
    trainer.optim_step(AdamParams(learning_rate=0.5)).result()
    checkpoint = trainer.save_state("eval-source").result()

    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(json.dumps({"prompt": "hello ", "expected": ""}) + "\n")
    sampler = service.create_sampling_client(model_path=checkpoint.path)
    report = run_eval(
        sampler,
        EvalSpec(
            name="smoke",
            dataset=str(dataset),
            scorer="contains",
            max_samples=1,
            sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
            checkpoint_path=checkpoint.path,
        ),
        store=RunStore(tmp_path / "checkpoints"),
    )
    export = export_checkpoint("local", checkpoint.path, output=str(tmp_path / "exported"))

    assert report.id.startswith("eval-")
    assert RunStore(tmp_path / "checkpoints").list_evals()[0]["id"] == report.id
    assert Path(export.output_path, "export.json").exists()
    assert "local" in list_exporters()
    service.close()


def test_gguf_and_ollama_exports_without_tooling_emit_plans(tmp_path, monkeypatch):
    monkeypatch.delenv("RAY_UNSLOTH_LLAMA_CPP", raising=False)
    service = ServiceClient(config=_config(tmp_path))
    trainer = service.create_lora_training_client()
    checkpoint = trainer.save_state("gguf-source").result()

    gguf = export_checkpoint("gguf", checkpoint.path, output=str(tmp_path / "gguf"))
    ollama = export_checkpoint("ollama", checkpoint.path, output=str(tmp_path / "ollama"))
    service.close()

    assert Path(gguf.output_path, "export-plan.json").exists()
    assert any("llama.cpp checkout not found" in note for note in gguf.notes)
    modelfile = Path(ollama.output_path, "Modelfile").read_text()
    assert modelfile.startswith("FROM ") and "ADAPTER" in modelfile


def test_config_plugin_registers_scorer_and_exporter(tmp_path):
    config = load_config(_config(tmp_path, plugins=["examples.sample_plugin"]))
    service = ServiceClient(config=config)
    trainer = service.create_lora_training_client()
    checkpoint = trainer.save_state("plugin-source").result()
    report = export_checkpoint("metadata-card", checkpoint.path, output=str(tmp_path / "card"))

    assert Path(report.output_path, "metadata-card.json").exists()
    service.close()


def test_cli_run_eval_export_and_runs(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(tmp_path)))

    assert cli_main(["--config", str(config_path), "validate-config"]) == 0
    capsys.readouterr()
    assert cli_main(["--config", str(config_path), "run", "--steps", "1", "--checkpoint-name", "cli-state"]) == 0
    run_output = json.loads(capsys.readouterr().out)
    checkpoint = run_output["checkpoint"]
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(json.dumps({"prompt": "ray", "expected": ""}) + "\n")

    assert cli_main(["--config", str(config_path), "eval", checkpoint, str(dataset), "--max-samples", "1"]) == 0
    assert cli_main(["export", checkpoint, "--target", "hf", "--output", str(tmp_path / "hf")]) == 0
    assert cli_main(["--config", str(config_path), "runs", "--json"]) == 0


def test_ui_api_uses_real_store(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ray_unsloth.ui.server import create_app

    config = RuntimeConfig.from_dict(_config(tmp_path))
    store = RunStore(config.checkpoint_root)
    run = store.create_run(provider="fake", base_model="Test/Test-1B")
    store.append_metrics(run.id, {"event": "test", "loss": 1.0})
    client = TestClient(create_app(config))

    assert client.get("/api/summary").json()["runs"] == 1
    assert client.get(f"/api/runs/{run.id}").json()["metrics"][0]["loss"] == 1.0
    assert client.get("/api/providers").json()[0]["name"]
