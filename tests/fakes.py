from __future__ import annotations

from ray_unsloth import AdamParams, ModelInput, SamplingParams
from ray_unsloth.clients.sampling import SamplingClient
from ray_unsloth.types import (
    CheckpointRef,
    ForwardBackwardOutput,
    GeneratedSequence,
    OptimStepResult,
    SampleResponse,
    SaveWeightsForSamplerResponse,
    TensorData,
    TrainingClientInfo,
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

    def get_info(self):
        return TrainingClientInfo(
            session_id="train",
            base_model="Qwen/Qwen3.5-4B",
            lora_rank=16,
        )

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


__all__ = [
    "FakeModalRuntime",
    "FakeRuntimeSession",
    "FakeSamplerActor",
    "FakeService",
    "FakeTrainerActor",
]
