import asyncio
import sys
from types import SimpleNamespace

import pytest

from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams
from ray_unsloth.checkpoints import base_manifest, write_manifest
from ray_unsloth.clients import service as service_module
from ray_unsloth.clients._remote import call, call_async
from ray_unsloth.clients.sampling import SamplingClient
from ray_unsloth.clients.service import ServiceClient
from ray_unsloth.clients.training import TrainingClient
from ray_unsloth.config import ModelConfig
from ray_unsloth.errors import CheckpointError
from ray_unsloth.runtime.modal import ModalActorHandle, ModalSession
from ray_unsloth.types import TensorData, TrainingClientInfo
from tests.fakes import FakeModalRuntime, FakeRuntimeSession, FakeSamplerActor, FakeService, FakeTrainerActor


def test_training_client_primitives_return_futures():
    client = TrainingClient(session_id="train", actor=FakeTrainerActor(), service=FakeService())
    datum = Datum(model_input=ModelInput.from_ints([1, 2]))

    assert client.forward_backward([datum]).result().loss == 1.25
    assert client.optim_step(AdamParams(learning_rate=1e-4)).result().step == 1


def test_training_client_async_aliases_and_create_sampling_client():
    service = FakeService()
    client = TrainingClient(session_id="train", actor=FakeTrainerActor(), service=service)
    datum = Datum(model_input=ModelInput.from_ints([1, 2]))

    assert client.forward_backward_async([datum]).result().loss == 1.25
    sampler = client.create_sampling_client("/tmp/sampler")

    assert isinstance(sampler, SamplingClient)
    assert service.model_path == "/tmp/sampler"


def test_training_client_get_info_async_matches_sync():
    client = TrainingClient(session_id="train", actor=FakeTrainerActor(), service=FakeService())

    info = asyncio.run(client.get_info_async())

    assert isinstance(info, TrainingClientInfo)
    assert info.session_id == "train"
    assert info.base_model == "Qwen/Qwen3.5-4B"
    assert info.lora_rank == 16
    assert info == client.get_info()


def test_tinker_first_sft_training_primitive_loop_shape(monkeypatch):
    """Exercise the official cookbook loop shape against the public clients."""

    monkeypatch.setattr(service_module, "RaySession", FakeRuntimeSession)

    async def run_loop():
        service_client = ServiceClient(config={})
        training_client = await service_client.create_lora_training_client_async(
            base_model="Qwen/Qwen3.5-4B",
            rank=16,
        )
        training_data = [
            Datum(
                model_input=ModelInput.from_ints([101, 102]),
                loss_fn_inputs={
                    "labels": TensorData(data=[-100, 102], dtype="int64", shape=[2]),
                    "weights": TensorData(data=[0.0, 1.0], dtype="float32", shape=[2]),
                },
            )
        ]
        losses = []
        for _step in range(2):
            fwdbwd_future = await training_client.forward_backward_async(
                training_data,
                "cross_entropy",
            )
            optim_future = await training_client.optim_step_async(AdamParams(learning_rate=0.0002))
            fwdbwd_result = await fwdbwd_future.result_async()
            await optim_future.result_async()
            logprobs = [value for output in fwdbwd_result.loss_fn_outputs for value in output["logprobs"].tolist()]
            weights = [value for datum in training_data for value in datum.loss_fn_inputs["weights"].tolist()]
            losses.append(-sum(lp * weight for lp, weight in zip(logprobs, weights, strict=False)) / sum(weights))
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name="tinker-tinker-sft",
        )
        sample_future = await sampling_client.sample_async(
            ModelInput.from_ints([101]),
            num_samples=1,
            sampling_params=SamplingParams(max_tokens=2, temperature=0.7),
        )
        sample = await sample_future.result_async()
        return losses, sample, training_client

    losses, sample, training_client = asyncio.run(run_loop())

    assert losses == [1.25, 1.25]
    assert sample.sequences[0].text == "trained"
    assert training_client._actor.saved_sampler_path == "tinker-tinker-sft"


def test_service_client_close_delegates_to_runtime_session(monkeypatch):
    monkeypatch.setattr(service_module, "RaySession", FakeRuntimeSession)

    service_client = ServiceClient(config={})
    service_client.close()

    assert service_client._session.closed is True


def test_save_weights_and_get_sampling_client_reuses_training_actor():
    service = FakeService()
    actor = FakeTrainerActor()
    client = TrainingClient(session_id="train", actor=actor, service=service)

    sampler = client.save_weights_and_get_sampling_client()

    assert isinstance(sampler, SamplingClient)
    assert (
        sampler.sample(ModelInput.from_ints([1]), sampling_params=SamplingParams(max_tokens=2))
        .result()
        .sequences[0]
        .text
        == "trained"
    )
    assert service.model_path is None
    assert actor.saved_sampler_path is None


def test_create_live_sampling_client_reuses_training_actor_without_saving():
    service = FakeService()
    actor = FakeTrainerActor()
    client = TrainingClient(session_id="train", actor=actor, service=service)

    sampler = client.create_live_sampling_client(name="policy")

    assert sampler.session_id == "train-policy"
    assert (
        sampler.sample(ModelInput.from_ints([1]), sampling_params=SamplingParams(max_tokens=2))
        .result()
        .sequences[0]
        .text
        == "trained"
    )
    assert actor.saved_sampler_path is None


def test_save_weights_and_get_sampling_client_uses_service_for_replicas():
    service = FakeService()
    client = TrainingClient(session_id="train", actor=FakeTrainerActor(), service=service)

    sampler = client.save_weights_and_get_sampling_client(replicas=2)

    assert isinstance(sampler, SamplingClient)
    assert service.model_path == "/tmp/sampler"


def test_sampling_client_round_robin_and_sample_future():
    actors = [FakeSamplerActor(), FakeSamplerActor()]
    client = SamplingClient(session_id="sample", actors=actors)

    response = client.sample(
        ModelInput.from_ints([1]),
        sampling_params=SamplingParams(max_tokens=2),
    ).result()
    assert response.sequences[0].text == "ok"
    client.compute_logprobs(ModelInput.from_ints([1, 2])).result()

    assert actors[0].calls == 1


def test_service_client_selects_modal_session(monkeypatch):
    monkeypatch.setattr(service_module, "ModalSession", FakeRuntimeSession)

    client = ServiceClient({"modal": {"enabled": True}})

    assert isinstance(client._session, FakeRuntimeSession)
    assert client.get_server_capabilities().features["runtime_backend"] == "modal"


def test_service_client_reports_multi_trainer_capacity(monkeypatch):
    monkeypatch.setattr(service_module, "RaySession", FakeRuntimeSession)

    client = ServiceClient(config={"resources": {"trainer_replicas": 4}})
    capabilities = client.get_server_capabilities()

    assert capabilities.supports_multi_trainer is True
    assert capabilities.max_concurrent_trainers == 4
    assert capabilities.features["trainer_replicas"] == 4


def test_service_client_advertises_implemented_losses(monkeypatch):
    monkeypatch.setattr(service_module, "RaySession", FakeRuntimeSession)

    client = ServiceClient(config={})
    losses = set(client.get_server_capabilities().features["losses"])
    assert losses == {"cross_entropy", "importance_sampling", "ppo", "cispo"}


def test_service_client_accepts_tinker_signature_and_metadata(monkeypatch):
    monkeypatch.setattr(service_module, "RaySession", FakeRuntimeSession)
    client = ServiceClient(user_metadata={"owner": "test"}, project_id="project", config={})

    training = client.create_lora_training_client(
        "base/model",
        rank=8,
        seed=123,
        train_mlp=False,
        train_attn=True,
        train_unembed=False,
        user_metadata={"run": "one"},
    )

    assert isinstance(training, TrainingClient)
    assert client._session.training_kwargs["base_model"] == "base/model"
    assert client._session.training_kwargs["lora_rank"] == 8
    assert client._session.training_kwargs["seed"] == 123
    assert client._session.training_kwargs["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert client._session.training_kwargs["metadata"] == {
        "project_id": "project",
        "owner": "test",
        "run": "one",
    }


def test_service_client_uses_configured_lora_targets_by_default(monkeypatch):
    monkeypatch.setattr(service_module, "RaySession", FakeRuntimeSession)
    client = ServiceClient(config={"lora": {"target_modules": ["q_proj", "v_proj"]}})

    client.create_lora_training_client()

    assert client._session.training_kwargs["target_modules"] is None


def test_service_client_explicit_flags_override_configured_lora_targets(monkeypatch):
    monkeypatch.setattr(service_module, "RaySession", FakeRuntimeSession)
    client = ServiceClient(config={"lora": {"target_modules": ["q_proj", "v_proj"]}})

    client.create_lora_training_client(train_mlp=False, train_attn=True, train_unembed=False)

    assert client._session.training_kwargs["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_modal_actor_handle_matches_remote_call_shape():
    actor = ModalActorHandle(
        session=FakeModalRuntime(),
        actor_kind="trainer",
        session_id="train-1",
        init_kwargs={"session_id": "train-1"},
    )

    result = call(actor, "forward_backward", [Datum(model_input=ModelInput.from_ints([1]))]).result()

    assert result["actor_kind"] == "trainer"
    assert result["session_id"] == "train-1"
    assert result["method_name"] == "forward_backward"
    assert isinstance(result["args"][0][0], Datum)


def test_modal_session_invoke_uses_parameterized_actor_class():
    calls = []

    class FakeRemote:
        def remote(self, session_id, init_kwargs, method_name, args, kwargs):
            calls.append(("remote", session_id, init_kwargs, method_name, args, kwargs))
            return "ok"

    class FakeParameterizedActor:
        invoke = FakeRemote()

    class FakeActorClass:
        def __call__(self, **kwargs):
            calls.append(("construct", kwargs))
            return FakeParameterizedActor()

    session = ModalSession.__new__(ModalSession)
    session._actor_cls = FakeActorClass()

    result = ModalSession.invoke(
        session,
        actor_kind="trainer",
        session_id="train-1",
        pool_id="shared-pool",
        replica_index=2,
        init_kwargs={"session_id": "train-1"},
        method_name="get_info",
        args=(),
        kwargs={},
    )

    assert result == "ok"
    assert calls[0] == (
        "construct",
        {"actor_kind": "trainer", "pool_id": "shared-pool", "replica_index": 2},
    )
    assert calls[1][1] == "train-1"
    assert calls[1][2] == {"session_id": "train-1"}


def test_modal_actor_handle_async_uses_async_invoke_shape():
    actor = ModalActorHandle(
        session=FakeModalRuntime(),
        actor_kind="trainer",
        session_id="train-1",
        init_kwargs={"session_id": "train-1"},
    )

    async def run_call():
        future = await call_async(actor, "forward_backward", [Datum(model_input=ModelInput.from_ints([1]))])
        return await future.result_async()

    result = asyncio.run(run_call())

    assert result["async"] is True
    assert result["actor_kind"] == "trainer"
    assert result["method_name"] == "forward_backward"


def test_modal_actor_handle_get_tokenizer_loads_locally(monkeypatch):
    calls = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return "tokenizer"

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    actor = ModalActorHandle(
        session=FakeModalRuntime(),
        actor_kind="trainer",
        session_id="train-1",
        init_kwargs={"model_config": ModelConfig(base_model="test/model", trust_remote_code=False)},
    )

    assert call(actor, "get_tokenizer").result() == "tokenizer"
    assert calls == [(("test/model",), {"trust_remote_code": False})]


RESTORE_METHODS = [
    "create_training_client_from_state",
    "create_training_client_from_state_with_optimizer",
]

RESTORE_CONFIG = {
    "model": {"base_model": "base/model"},
    "lora": {"rank": 8, "target_modules": ["q_proj", "v_proj"]},
}


def _write_checkpoint(path, *, base_model="base/model", rank=8, target_modules=("q_proj", "v_proj")):
    write_manifest(
        path,
        base_manifest(
            kind="training_state",
            step=1,
            base_model=base_model,
            lora={"rank": rank, "target_modules": list(target_modules)},
            has_optimizer=True,
        ),
    )
    return str(path)


@pytest.mark.parametrize("method_name", RESTORE_METHODS)
def test_restore_proceeds_when_manifest_matches_config(monkeypatch, tmp_path, method_name):
    monkeypatch.setattr(service_module, "RaySession", FakeRuntimeSession)
    client = ServiceClient(config=RESTORE_CONFIG)
    path = _write_checkpoint(tmp_path / "checkpoint")

    training = getattr(client, method_name)(path)

    assert isinstance(training, TrainingClient)
    assert client._session.training_kwargs["model_path"] == path


@pytest.mark.parametrize("method_name", RESTORE_METHODS)
def test_restore_raises_on_base_model_mismatch(monkeypatch, tmp_path, method_name):
    monkeypatch.setattr(service_module, "RaySession", FakeRuntimeSession)
    client = ServiceClient(config=RESTORE_CONFIG)
    path = _write_checkpoint(tmp_path / "checkpoint", base_model="other/model")

    with pytest.raises(CheckpointError) as exc_info:
        getattr(client, method_name)(path)

    assert "base_model mismatch" in str(exc_info.value)
    # No actor was created because validation fails before session work.
    assert client._session.training_kwargs is None


@pytest.mark.parametrize("method_name", RESTORE_METHODS)
def test_restore_raises_on_rank_mismatch(monkeypatch, tmp_path, method_name):
    monkeypatch.setattr(service_module, "RaySession", FakeRuntimeSession)
    client = ServiceClient(config=RESTORE_CONFIG)
    path = _write_checkpoint(tmp_path / "checkpoint", rank=4)

    with pytest.raises(CheckpointError) as exc_info:
        getattr(client, method_name)(path)

    message = str(exc_info.value)
    assert "lora.rank" in message
    assert "rank 4" in message
    assert client._session.training_kwargs is None


@pytest.mark.parametrize("method_name", RESTORE_METHODS)
def test_restore_raises_on_missing_manifest(monkeypatch, tmp_path, method_name):
    monkeypatch.setattr(service_module, "RaySession", FakeRuntimeSession)
    client = ServiceClient(config=RESTORE_CONFIG)
    manifestless = tmp_path / "manifestless"
    manifestless.mkdir()

    with pytest.raises(CheckpointError):
        getattr(client, method_name)(str(manifestless))

    assert client._session.training_kwargs is None


@pytest.mark.parametrize("method_name", RESTORE_METHODS)
def test_restore_defers_validation_for_remote_paths(monkeypatch, method_name):
    """A checkpoint path not readable on this machine (e.g. a Modal volume path)
    skips client-side pre-validation; the engine validates authoritatively."""
    monkeypatch.setattr(service_module, "RaySession", FakeRuntimeSession)
    client = ServiceClient(config=RESTORE_CONFIG)

    getattr(client, method_name)("/checkpoints/lives-on-a-remote-volume")

    assert client._session.training_kwargs["model_path"] == "/checkpoints/lives-on-a-remote-volume"


def test_restore_rank_override_validates_against_override(monkeypatch, tmp_path):
    monkeypatch.setattr(service_module, "RaySession", FakeRuntimeSession)
    client = ServiceClient(config=RESTORE_CONFIG)
    # Checkpoint saved at rank 4; default config rank is 8, but the rank override matches.
    path = _write_checkpoint(tmp_path / "checkpoint", rank=4)

    training = client.create_training_client_from_state(path, rank=4)

    assert isinstance(training, TrainingClient)
    assert client._session.training_kwargs["lora_rank"] == 4
