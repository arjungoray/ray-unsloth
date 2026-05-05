import pytest

from ray_unsloth import AdamParams, Datum, ModelInput
from ray_unsloth.config import RuntimeConfig
from ray_unsloth.runtime.modal import session as modal_session
from ray_unsloth.runtime.modal.session import ModalSession
from ray_unsloth.runtime.ray import session as ray_session
from ray_unsloth.runtime.ray.distributed_trainer import DistributedTrainerCoordinator
from ray_unsloth.runtime.ray.trainer_actor import TrainerActorImpl
from ray_unsloth.types import (
    CheckpointRef,
    ForwardBackwardOutput,
    ForwardOutput,
    OptimStepResult,
    SaveWeightsForSamplerResponse,
    TensorData,
)


class FakeDistributedWorker:
    def __init__(self, rank: int, step: int = 1):
        self.rank = rank
        self.step = step
        self.calls: list[str] = []

    def get_tokenizer(self):
        return f"tokenizer-{self.rank}"

    def get_info(self):
        return f"info-{self.rank}"

    def register_custom_loss(self, name, loss_fn):
        del loss_fn
        self.calls.append(f"register:{name}")

    def compute_logprobs(self, prompt):
        del prompt
        return [None, -0.5]

    def sample(self, *args, **kwargs):
        del args, kwargs
        return "sample"

    def get_base_model(self):
        return "base/model"

    def forward_indexed(self, indexed_data, loss_fn="cross_entropy"):
        assert loss_fn == "cross_entropy"
        indexes = [index for index, _datum in indexed_data]
        outputs = [
            {
                "logprobs": TensorData(
                    data=[float(index)],
                    dtype="float32",
                    shape=[1],
                )
            }
            for index in indexes
        ]
        loss = float(sum(index + 1 for index in indexes))
        return (
            ForwardOutput(
                loss=loss,
                metrics={"loss": loss, "loss:sum": loss, "rank": float(self.rank)},
                loss_fn_outputs=outputs,
            ),
            list(zip(indexes, outputs)),
        )

    def forward_backward_indexed(self, indexed_data, loss_fn="cross_entropy", loss_fn_config=None):
        assert loss_fn_config == {"clip": 1}
        forward, indexed_outputs = self.forward_indexed(indexed_data, loss_fn=loss_fn)
        return (
            ForwardBackwardOutput(
                loss=forward.loss,
                metrics=forward.metrics,
                loss_fn_outputs=forward.loss_fn_outputs,
            ),
            indexed_outputs,
        )

    def forward_backward_custom_indexed(self, indexed_data, loss_fn, loss_fn_config=None):
        del indexed_data, loss_fn, loss_fn_config
        loss = float(self.rank + 1)
        return ForwardBackwardOutput(loss=loss, metrics={"loss": loss, "rank": float(self.rank)}), []

    def optim_step(self, adam_params):
        assert isinstance(adam_params, AdamParams)
        return OptimStepResult(step=self.step, metrics={"step": float(self.step), "rank": float(self.rank)})

    def save_state(self, path=None):
        self.calls.append(f"save_state:{path}")
        if self.rank == 0:
            return CheckpointRef(path=path or "/tmp/state", step=self.step)
        return None

    def save_state_with_optimizer(self, path=None):
        self.calls.append(f"save_state_with_optimizer:{path}")
        if self.rank == 0:
            return CheckpointRef(path=path or "/tmp/state-opt", step=self.step, has_optimizer=True)
        return None

    def load_state(self, path):
        self.calls.append(f"load_state:{path}")

    def load_state_with_optimizer(self, path):
        self.calls.append(f"load_state_with_optimizer:{path}")

    def save_weights_for_sampler(self, path=None):
        self.calls.append(f"save_weights_for_sampler:{path}")
        if self.rank == 0:
            checkpoint = CheckpointRef(path=path or "/tmp/sampler", step=self.step)
            return SaveWeightsForSamplerResponse(path=checkpoint.path, checkpoint=checkpoint)
        return None


def _data(count: int) -> list[Datum]:
    return [Datum(model_input=ModelInput.from_ints([index])) for index in range(count)]


def test_runtime_config_parses_and_validates_distributed():
    config = RuntimeConfig.from_dict(
        {
            "distributed": {
                "enabled": True,
                "mode": "ddp",
                "num_nodes": 1,
                "gpus_per_node": 4,
                "backend": "nccl",
            }
        }
    )

    assert config.distributed.enabled is True
    assert config.distributed.mode == "ddp"
    assert config.distributed.gpus_per_node == 4
    assert config.distributed.placement_strategy == "STRICT_PACK"
    assert RuntimeConfig.from_dict({"distributed": {"mode": "ddp"}}).distributed.enabled is True

    with pytest.raises(ValueError, match="mode must be 'ddp'"):
        RuntimeConfig.from_dict({"distributed": {"enabled": True, "mode": "fsdp"}})
    with pytest.raises(ValueError, match="num_nodes == 1"):
        RuntimeConfig.from_dict({"distributed": {"enabled": True, "mode": "ddp", "num_nodes": 2}})
    with pytest.raises(ValueError, match="gpus_per_node"):
        RuntimeConfig.from_dict({"distributed": {"enabled": True, "mode": "ddp", "gpus_per_node": 0}})


def test_coordinator_public_methods_match_trainer_actor_impl():
    coordinator_methods = {
        name
        for name in dir(DistributedTrainerCoordinator(workers=[FakeDistributedWorker(0)]))
        if not name.startswith("_") and callable(getattr(DistributedTrainerCoordinator, name, None))
    }
    trainer_methods = {
        name
        for name in dir(TrainerActorImpl)
        if not name.startswith("_") and callable(getattr(TrainerActorImpl, name, None))
    }

    assert trainer_methods <= coordinator_methods


def test_coordinator_shards_and_restores_loss_outputs_in_original_order():
    coordinator = DistributedTrainerCoordinator(
        workers=[FakeDistributedWorker(0), FakeDistributedWorker(1)],
        world_size=2,
    )

    result = coordinator.forward(_data(5))

    assert result.loss == 15.0
    assert result.metrics["loss"] == 15.0
    assert result.metrics["loss:sum"] == 15.0
    assert result.metrics["rank"] == 1.0
    assert [output["logprobs"].tolist()[0] for output in result.loss_fn_outputs] == [0, 1, 2, 3, 4]


def test_coordinator_forward_backward_requires_one_datum_per_rank_and_aggregates():
    coordinator = DistributedTrainerCoordinator(
        workers=[FakeDistributedWorker(0), FakeDistributedWorker(1), FakeDistributedWorker(2)],
        world_size=3,
    )

    with pytest.raises(ValueError, match="at least one datum per rank"):
        coordinator.forward_backward(_data(2))

    result = coordinator.forward_backward(_data(3), loss_fn_config={"clip": 1})

    assert result.loss == 6.0
    assert [output["logprobs"].tolist()[0] for output in result.loss_fn_outputs] == [0, 1, 2]


def test_coordinator_custom_loss_and_registration_run_on_all_ranks():
    workers = [FakeDistributedWorker(0), FakeDistributedWorker(1)]
    coordinator = DistributedTrainerCoordinator(workers=workers, world_size=2)

    coordinator.register_custom_loss("custom", lambda outputs, data, config: (outputs, {}))
    result = coordinator.forward_backward_custom(_data(2), "custom")

    assert [worker.calls for worker in workers] == [["register:custom"], ["register:custom"]]
    assert result.loss == 3.0
    assert result.metrics["rank"] == 1.0


def test_coordinator_optim_step_returns_rank0_canonical_result():
    coordinator = DistributedTrainerCoordinator(
        workers=[FakeDistributedWorker(0, step=7), FakeDistributedWorker(1, step=7)],
        world_size=2,
    )

    result = coordinator.optim_step(AdamParams(learning_rate=1e-4))

    assert result.step == 7
    assert result.metrics["rank"] == 0.0


def test_coordinator_save_returns_rank0_result_and_calls_all_workers():
    workers = [FakeDistributedWorker(0), FakeDistributedWorker(1)]
    coordinator = DistributedTrainerCoordinator(workers=workers, world_size=2)

    state = coordinator.save_state("/tmp/state")
    sampler = coordinator.save_weights_for_sampler("/tmp/sampler")

    assert state.path == "/tmp/state"
    assert sampler.path == "/tmp/sampler"
    assert workers[0].calls == ["save_state:/tmp/state", "save_weights_for_sampler:/tmp/sampler"]
    assert workers[1].calls == ["save_state:/tmp/state", "save_weights_for_sampler:/tmp/sampler"]


def test_ray_session_default_training_actor_path_still_uses_trainer_actor(monkeypatch):
    factory_calls = {}

    class FakeTrainerActorFactory:
        def options(self, **options):
            factory_calls["options"] = options
            return self

        def remote(self, **kwargs):
            factory_calls["remote"] = kwargs
            return "trainer-actor"

    monkeypatch.setattr(ray_session, "TrainerActor", FakeTrainerActorFactory())
    session = ray_session.RaySession.__new__(ray_session.RaySession)
    session.config = RuntimeConfig.from_dict({})
    session._placement_group = None
    session.training_actors = {}

    session_id, actor = ray_session.RaySession.create_training_actor(session, base_model="base/model")

    assert session_id.startswith("train-")
    assert actor == "trainer-actor"
    assert factory_calls["options"]["num_gpus"] == 1.0
    assert factory_calls["remote"]["model_config"].base_model == "base/model"


def test_modal_training_actor_carries_distributed_config_and_gpu_string():
    session = ModalSession.__new__(ModalSession)
    session.config = RuntimeConfig.from_dict(
        {
            "distributed": {"enabled": True, "mode": "ddp", "gpus_per_node": 4},
            "modal": {"enabled": True, "gpu": "H100:4"},
        }
    )
    session.training_actors = {}

    _session_id, actor = ModalSession.create_training_actor(session)

    assert session.config.modal.gpu == "H100:4"
    assert actor.init_kwargs["distributed_config"].enabled is True
    assert actor.init_kwargs["distributed_config"].gpus_per_node == 4


def test_modal_function_is_single_container_for_stateful_actor_registry():
    config = RuntimeConfig.from_dict(
        {
            "modal": {
                "gpu": "L4:2",
                "timeout": 123,
                "scaledown_window": 45,
                "volume_mount_path": "/checkpoints",
            }
        }
    )
    kwargs = modal_session._modal_function_kwargs(config, image="image", volume="volume")

    assert kwargs["gpu"] == "L4:2"
    assert kwargs["timeout"] == 123
    assert kwargs["scaledown_window"] == 45
    assert kwargs["volumes"] == {"/checkpoints": "volume"}
    assert kwargs["max_containers"] == 1


def test_modal_invoke_routes_ddp_trainers_to_coordinator(monkeypatch):
    class FakeCoordinator:
        def get_base_model(self):
            return "distributed-base"

    monkeypatch.setattr(modal_session, "_create_modal_distributed_trainer", lambda init_kwargs: FakeCoordinator())
    modal_session._ENGINE_REGISTRY.clear()
    config = RuntimeConfig.from_dict({"distributed": {"enabled": True, "mode": "ddp", "gpus_per_node": 2}})

    result = modal_session._modal_invoke_impl(
        "trainer",
        "train-ddp",
        {"distributed_config": config.distributed},
        "get_base_model",
        (),
        {},
    )

    assert result == "distributed-base"
    assert isinstance(modal_session._ENGINE_REGISTRY[("trainer", "train-ddp")], FakeCoordinator)
