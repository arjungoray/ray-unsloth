import asyncio
import sys
from types import SimpleNamespace

from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams
from ray_unsloth.clients import service as service_module
from ray_unsloth.clients.sampling import SamplingClient
from ray_unsloth.clients.service import ServiceClient
from ray_unsloth.clients.training import TrainingClient
from ray_unsloth.clients._remote import call, call_async
from ray_unsloth.config import ModelConfig
from ray_unsloth.runtime.modal import ModalActorHandle, ModalSession
from ray_unsloth.types import (
    CheckpointRef,
    ForwardBackwardOutput,
    GeneratedSequence,
    OptimStepResult,
    SampleResponse,
    SaveWeightsForSamplerResponse,
    TensorData,
)


class FakeTrainerActor:
    def __init__(self):
        self.metadata = None
        self.saved_sampler_path = None

    def forward_backward(self, data, loss_fn="cross_entropy", loss_fn_config=None):
        assert loss_fn == "cross_entropy"
        assert loss_fn_config is None
        if "weights" in data[0].loss_fn_inputs:
            assert data[0].loss_fn_inputs["weights"].tolist() == [0.0, 1.0]
        return ForwardBackwardOutput(
            loss=1.25,
            loss_fn_outputs=[
                {
                    "logprobs": TensorData(
                        data=[0.0, -1.25],
                        dtype="float32",
                        shape=[2],
                    )
                }
            ],
        )

    def optim_step(self, adam_params):
        assert isinstance(adam_params, AdamParams)
        return OptimStepResult(step=1)

    def save_weights_for_sampler(self, path=None):
        self.saved_sampler_path = path
        checkpoint = CheckpointRef(path="/tmp/sampler", step=1)
        return SaveWeightsForSamplerResponse(path=checkpoint.path, checkpoint=checkpoint)

    def sample(
        self,
        prompt,
        num_samples=1,
        sampling_params=None,
        include_prompt_logprobs=False,
        topk_prompt_logprobs=0,
    ):
        del prompt, num_samples, sampling_params, include_prompt_logprobs, topk_prompt_logprobs
        return SampleResponse(sequences=[GeneratedSequence(tokens=[1, 2], text="trained")])


class FakeSamplerActor:
    def __init__(self):
        self.calls = 0

    def sample(
        self,
        prompt,
        num_samples=1,
        sampling_params=None,
        include_prompt_logprobs=False,
        topk_prompt_logprobs=0,
    ):
        assert include_prompt_logprobs is False
        assert topk_prompt_logprobs == 0
        self.calls += 1
        assert isinstance(prompt, ModelInput)
        assert isinstance(sampling_params, SamplingParams)
        return SampleResponse(sequences=[GeneratedSequence(tokens=[1, 2], text="ok")])

    def compute_logprobs(self, prompt):
        del prompt
        return [None, -0.5]

    def get_base_model(self):
        return "base"


class FakeService:
    def __init__(self):
        self.model_path = None

    def create_sampling_client(self, *, model_path=None, retry_config=None, replicas=None):
        del retry_config, replicas
        self.model_path = model_path
        return SamplingClient(session_id="sample", actors=[FakeSamplerActor()])


class FakeRuntimeSession:
    def __init__(self, config):
        self.config = config
        self.training_kwargs = None
        self.closed = False

    def create_training_actor(self, **kwargs):
        self.training_kwargs = kwargs
        return "train", FakeTrainerActor()

    def create_sampler_actors(self, **kwargs):
        del kwargs
        return "sample", [FakeSamplerActor()]

    def close(self):
        self.closed = True


class FakeModalRuntime:
    def invoke(self, **kwargs):
        return {
            "actor_kind": kwargs["actor_kind"],
            "session_id": kwargs["session_id"],
            "pool_id": kwargs["pool_id"],
            "method_name": kwargs["method_name"],
            "args": kwargs["args"],
            "kwargs": kwargs["kwargs"],
        }

    async def invoke_async(self, **kwargs):
        result = self.invoke(**kwargs)
        result["async"] = True
        return result


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
            logprobs = [
                value
                for output in fwdbwd_result.loss_fn_outputs
                for value in output["logprobs"].tolist()
            ]
            weights = [
                value
                for datum in training_data
                for value in datum.loss_fn_inputs["weights"].tolist()
            ]
            losses.append(-sum(lp * weight for lp, weight in zip(logprobs, weights)) / sum(weights))
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
    assert sampler.sample(ModelInput.from_ints([1]), sampling_params=SamplingParams(max_tokens=2)).result().sequences[0].text == "trained"
    assert service.model_path is None
    assert actor.saved_sampler_path is None


def test_create_live_sampling_client_reuses_training_actor_without_saving():
    service = FakeService()
    actor = FakeTrainerActor()
    client = TrainingClient(session_id="train", actor=actor, service=service)

    sampler = client.create_live_sampling_client(name="policy")

    assert sampler.session_id == "train-policy"
    assert sampler.sample(ModelInput.from_ints([1]), sampling_params=SamplingParams(max_tokens=2)).result().sequences[0].text == "trained"
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
