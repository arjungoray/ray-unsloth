"""Single-node DDP trainer coordination for Ray-backed Unsloth workers."""

from __future__ import annotations

import os
import socket
from typing import Any

from ray_unsloth.config import LoRAConfig, ModelConfig, SpeedConfig
from ray_unsloth.runtime.unsloth import UnslothEngine
from ray_unsloth.types import (
    AdamParams,
    CustomLoss,
    Datum,
    ForwardBackwardOutput,
    ForwardOutput,
    ModelInput,
    OptimStepResult,
    SamplingParams,
)


def _is_object_ref(value: Any) -> bool:
    return value.__class__.__name__ == "ObjectRef"


def _resolve(value: Any) -> Any:
    if _is_object_ref(value):
        import ray

        return ray.get(value)
    result = getattr(value, "result", None)
    if callable(result):
        return result()
    get = getattr(value, "get", None)
    if callable(get):
        return get()
    return value


def _resolve_many(values: list[Any]) -> list[Any]:
    refs = [value for value in values if _is_object_ref(value)]
    if refs and len(refs) == len(values):
        import ray

        return ray.get(values)
    return [_resolve(value) for value in values]


def _call(actor: Any, method_name: str, *args, **kwargs) -> Any:
    method = getattr(actor, method_name)
    remote = getattr(method, "remote", None)
    if callable(remote):
        return remote(*args, **kwargs)
    return method(*args, **kwargs)


def _sum_metrics(outputs: list[Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for output in outputs:
        for key, value in getattr(output, "metrics", {}).items():
            metrics[key] = metrics.get(key, 0.0) + float(value)
    return metrics


def _ordered_loss_outputs(indexed_outputs: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [output for _index, output in sorted(indexed_outputs, key=lambda item: item[0])]


def _shard_indexed_data(data: list[Datum], world_size: int) -> list[list[tuple[int, Datum]]]:
    if len(data) < world_size:
        raise ValueError(
            "DDP training requires at least one datum per rank; "
            f"got {len(data)} datum(s) for world_size={world_size}."
        )
    shards: list[list[tuple[int, Datum]]] = [[] for _ in range(world_size)]
    for index, datum in enumerate(data):
        shards[index % world_size].append((index, datum))
    return shards


def _free_tcp_endpoint() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    try:
        host = socket.gethostbyname(socket.gethostname())
    except OSError:
        host = "127.0.0.1"
    return f"tcp://{host}:{port}"


class DistributedTrainerWorker:
    """One DDP worker process that owns one Unsloth engine replica."""

    def __init__(
        self,
        *,
        session_id: str,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        checkpoint_root: str,
        speed_config: SpeedConfig | None = None,
        rank: int,
        world_size: int,
        backend: str,
        model_path: str | None = None,
        with_optimizer: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.model_config = model_config
        self.lora_config = lora_config
        self.checkpoint_root = checkpoint_root
        self.speed_config = speed_config
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.model_path = model_path
        self.with_optimizer = with_optimizer
        self.metadata = metadata or {}
        self.engine: UnslothEngine | None = None

    def get_process_group_endpoint(self) -> str:
        if self.rank != 0:
            raise RuntimeError("Only rank 0 can create a DDP process-group endpoint.")
        return _free_tcp_endpoint()

    def initialize(self, init_method: str) -> bool:
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = "0"
        self.engine = UnslothEngine(
            session_id=self.session_id,
            model_config=self.model_config,
            lora_config=self.lora_config,
            checkpoint_root=self.checkpoint_root,
            speed_config=self.speed_config,
            model_path=self.model_path,
            with_optimizer=self.with_optimizer,
            metadata=self.metadata,
            rank=self.rank,
            local_rank=0,
            world_size=self.world_size,
            backend=self.backend,
            init_method=init_method,
        )
        return True

    def _engine(self) -> UnslothEngine:
        if self.engine is None:
            raise RuntimeError("DistributedTrainerWorker has not been initialized.")
        return self.engine

    def get_tokenizer(self):
        return self._engine().get_tokenizer()

    def get_info(self):
        return self._engine().get_info()

    def register_custom_loss(self, name: str, loss_fn: CustomLoss) -> None:
        self._engine().register_custom_loss(name, loss_fn)

    def compute_logprobs(self, prompt):
        return self._engine().compute_logprobs(prompt)

    def sample(
        self,
        prompt: ModelInput | list[int],
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ):
        return self._engine().sample(
            prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
        )

    def get_base_model(self) -> str:
        return self._engine().model_config.base_model

    def forward_indexed(self, indexed_data: list[tuple[int, Datum]], loss_fn: str = "cross_entropy"):
        indexes = [index for index, _datum in indexed_data]
        output = self._engine().forward([datum for _index, datum in indexed_data], loss_fn=loss_fn)
        return output, list(zip(indexes, output.loss_fn_outputs))

    def forward_backward_indexed(
        self,
        indexed_data: list[tuple[int, Datum]],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, Any] | None = None,
    ):
        indexes = [index for index, _datum in indexed_data]
        output = self._engine().forward_backward(
            [datum for _index, datum in indexed_data],
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
        return output, list(zip(indexes, output.loss_fn_outputs))

    def forward_backward_custom_indexed(
        self,
        indexed_data: list[tuple[int, Datum]],
        loss_fn,
        loss_fn_config: dict[str, Any] | None = None,
    ):
        output = self._engine().forward_backward_custom(
            [datum for _index, datum in indexed_data],
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
        return output, []

    def optim_step(self, adam_params: AdamParams):
        return self._engine().optim_step(adam_params)

    def save_state(self, path: str | None = None):
        return self._engine().save_state(path=path, include_optimizer=False)

    def save_state_with_optimizer(self, path: str | None = None):
        return self._engine().save_state(path=path, include_optimizer=True)

    def load_state(self, path: str):
        return self._engine().load_state(path, with_optimizer=False)

    def load_state_with_optimizer(self, path: str):
        return self._engine().load_state_with_optimizer(path)

    def save_weights_for_sampler(self, path: str | None = None):
        return self._engine().save_weights_for_sampler(path=path)


class DistributedTrainerCoordinator:
    """Coordinator with the same public method surface as TrainerActorImpl."""

    def __init__(self, *, workers: list[Any], world_size: int | None = None) -> None:
        self.workers = workers
        self.world_size = world_size or len(workers)
        if self.world_size < 1:
            raise ValueError("DistributedTrainerCoordinator requires at least one worker.")

    def _rank0(self, method_name: str, *args, **kwargs):
        return _resolve(_call(self.workers[0], method_name, *args, **kwargs))

    def _all(self, method_name: str, args_by_rank: list[tuple[Any, ...]], kwargs: dict[str, Any] | None = None):
        kwargs = kwargs or {}
        calls = [
            _call(worker, method_name, *args_by_rank[rank], **kwargs)
            for rank, worker in enumerate(self.workers)
        ]
        return _resolve_many(calls)

    def _aggregate_forward(self, worker_results: list[tuple[Any, list[tuple[int, dict[str, Any]]]]], output_type):
        outputs = [result[0] for result in worker_results]
        indexed_outputs = [item for _output, indexed in worker_results for item in indexed]
        loss = sum(float(output.loss or 0.0) for output in outputs)
        metrics = _sum_metrics(outputs)
        metrics["loss"] = loss
        metrics["loss:sum"] = loss
        return output_type(
            loss=loss,
            metrics=metrics,
            loss_fn_outputs=_ordered_loss_outputs(indexed_outputs),
        )

    def get_tokenizer(self):
        return self._rank0("get_tokenizer")

    def get_info(self):
        return self._rank0("get_info")

    def register_custom_loss(self, name: str, loss_fn: CustomLoss) -> None:
        self._all("register_custom_loss", [(name, loss_fn) for _ in self.workers])

    def compute_logprobs(self, prompt):
        return self._rank0("compute_logprobs", prompt)

    def sample(
        self,
        prompt: ModelInput | list[int],
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ):
        return self._rank0(
            "sample",
            prompt,
            num_samples,
            sampling_params,
            include_prompt_logprobs,
            topk_prompt_logprobs,
        )

    def get_base_model(self) -> str:
        return self._rank0("get_base_model")

    def forward(self, data: list[Datum], loss_fn: str = "cross_entropy"):
        shards = _shard_indexed_data(data, self.world_size)
        results = self._all("forward_indexed", [(shard, loss_fn) for shard in shards])
        return self._aggregate_forward(results, ForwardOutput)

    def forward_backward(
        self,
        data: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, Any] | None = None,
    ):
        shards = _shard_indexed_data(data, self.world_size)
        results = self._all(
            "forward_backward_indexed",
            [(shard, loss_fn, loss_fn_config) for shard in shards],
        )
        return self._aggregate_forward(results, ForwardBackwardOutput)

    def forward_backward_custom(
        self,
        data: list[Datum],
        loss_fn,
        loss_fn_config: dict[str, Any] | None = None,
    ):
        shards = _shard_indexed_data(data, self.world_size)
        results = self._all(
            "forward_backward_custom_indexed",
            [(shard, loss_fn, loss_fn_config) for shard in shards],
        )
        outputs = [result[0] for result in results]
        loss = sum(float(output.loss) for output in outputs)
        metrics = _sum_metrics(outputs)
        metrics["loss"] = loss
        return ForwardBackwardOutput(loss=loss, metrics=metrics)

    def optim_step(self, adam_params: AdamParams):
        results: list[OptimStepResult] = self._all("optim_step", [(adam_params,) for _ in self.workers])
        rank0 = results[0]
        if any(result.step != rank0.step for result in results):
            raise RuntimeError(f"DDP worker optimizer steps diverged: {[result.step for result in results]}")
        return rank0

    def save_state(self, path: str | None = None):
        results = self._all("save_state", [(path,) for _ in self.workers])
        return results[0]

    def save_state_with_optimizer(self, path: str | None = None):
        results = self._all("save_state_with_optimizer", [(path,) for _ in self.workers])
        return results[0]

    def load_state(self, path: str):
        self._all("load_state", [(path,) for _ in self.workers])

    def load_state_with_optimizer(self, path: str):
        self._all("load_state_with_optimizer", [(path,) for _ in self.workers])

    def save_weights_for_sampler(self, path: str | None = None):
        results = self._all("save_weights_for_sampler", [(path,) for _ in self.workers])
        return results[0]


try:
    import ray

    DistributedTrainerWorkerActor = ray.remote(DistributedTrainerWorker)
    DistributedTrainerCoordinatorActor = ray.remote(DistributedTrainerCoordinator)
except ImportError:  # pragma: no cover - import-time fallback
    DistributedTrainerWorkerActor = DistributedTrainerWorker
    DistributedTrainerCoordinatorActor = DistributedTrainerCoordinator
