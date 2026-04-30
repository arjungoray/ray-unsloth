from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams
from ray_unsloth.clients.sampling import SamplingClient
from ray_unsloth.clients.training import TrainingClient
from ray_unsloth.types import (
    ForwardBackwardOutput,
    GeneratedSequence,
    OptimStepResult,
    SampleResponse,
    SaveWeightsForSamplerResponse,
    CheckpointRef,
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
