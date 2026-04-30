import sys
from types import SimpleNamespace

from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams
from ray_unsloth.clients import service as service_module
from ray_unsloth.clients.sampling import SamplingClient
from ray_unsloth.clients.service import ServiceClient
from ray_unsloth.clients.training import TrainingClient
from ray_unsloth.clients._remote import call
from ray_unsloth.config import ModelConfig
from ray_unsloth.runtime.modal import ModalActorHandle
from ray_unsloth.types import (
    CheckpointRef,
    ForwardBackwardOutput,
    GeneratedSequence,
    OptimStepResult,
    SampleResponse,
    SaveWeightsForSamplerResponse,
)


class FakeTrainerActor:
    def forward_backward(self, data, loss_fn="cross_entropy", loss_fn_config=None):
        assert loss_fn == "cross_entropy"
        assert loss_fn_config is None
        return ForwardBackwardOutput(loss=1.25)

    def optim_step(self, adam_params):
        assert isinstance(adam_params, AdamParams)
        return OptimStepResult(step=1)

    def save_weights_for_sampler(self, path=None):
        del path
        checkpoint = CheckpointRef(path="/tmp/sampler", step=1)
        return SaveWeightsForSamplerResponse(path=checkpoint.path, checkpoint=checkpoint)


class FakeSamplerActor:
    def __init__(self):
        self.calls = 0

    def sample(self, prompt, num_samples=1, sampling_params=None, **kwargs):
        del kwargs
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

    def create_sampling_client(self, *, model_path=None, replicas=None):
        del replicas
        self.model_path = model_path
        return SamplingClient(session_id="sample", actors=[FakeSamplerActor()])


class FakeRuntimeSession:
    def __init__(self, config):
        self.config = config


class FakeModalRuntime:
    def invoke(self, **kwargs):
        return {
            "actor_kind": kwargs["actor_kind"],
            "session_id": kwargs["session_id"],
            "method_name": kwargs["method_name"],
            "args": kwargs["args"],
            "kwargs": kwargs["kwargs"],
        }


def test_training_client_primitives_return_futures():
    client = TrainingClient(session_id="train", actor=FakeTrainerActor(), service=FakeService())
    datum = Datum(model_input=ModelInput.from_ints([1, 2]))

    assert client.forward_backward([datum]).result().loss == 1.25
    assert client.optim_step(AdamParams(learning_rate=1e-4)).result().step == 1


def test_save_weights_and_get_sampling_client_uses_service():
    service = FakeService()
    client = TrainingClient(session_id="train", actor=FakeTrainerActor(), service=service)

    sampler = client.save_weights_and_get_sampling_client()

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
