"""Ray session and actor registry."""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any

from ray_unsloth.config import LoRAConfig, ModelConfig, RuntimeConfig
from ray_unsloth.errors import RayUnavailableError
from ray_unsloth.runtime.ray.distributed_trainer import (
    DistributedTrainerCoordinatorActor,
    DistributedTrainerWorkerActor,
)
from ray_unsloth.runtime.ray.sampler_actor import SamplerActor
from ray_unsloth.runtime.ray.trainer_actor import TrainerActor


class RaySession:
    """Creates and tracks Ray actors for client sessions."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._placement_group = None
        self.training_actors: dict[str, Any] = {}
        self.sampler_actors: dict[str, list[Any]] = {}
        self._ray = self._init_ray()
        self._create_placement_group()

    def _init_ray(self):
        try:
            import ray
        except ImportError as exc:
            raise RayUnavailableError("Ray is required for ServiceClient runtime operations.") from exc
        if not ray.is_initialized():
            ray.init(
                address=self.config.ray.address,
                namespace=self.config.ray.namespace,
                ignore_reinit_error=self.config.ray.ignore_reinit_error,
            )
        return ray

    def _create_placement_group(self) -> None:
        if self.config.distributed.enabled:
            return
        resources = self.config.resources
        total_bundles = 1 + max(resources.sampler_replicas, 0)
        bundles = [
            {"GPU": resources.trainer_num_gpus, "CPU": resources.trainer_num_cpus},
            *[
                {"GPU": resources.sampler_num_gpus, "CPU": resources.sampler_num_cpus}
                for _ in range(total_bundles - 1)
            ],
        ]
        if not bundles:
            return
        try:
            from ray.util.placement_group import placement_group

            self._placement_group = placement_group(
                bundles,
                strategy=self.config.resources.placement_strategy,
                name=f"ray-unsloth-{uuid.uuid4().hex[:8]}",
            )
            self._ray.get(self._placement_group.ready(), timeout=120)
        except Exception:
            self._placement_group = None

    def _strategy(self, bundle_index: int, placement_group=None):
        placement_group = placement_group or self._placement_group
        if placement_group is None:
            return None
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        return PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=bundle_index,
        )

    def _create_distributed_placement_group(self, session_id: str):
        from ray.util.placement_group import placement_group

        resources = self.config.resources
        distributed = self.config.distributed
        bundles = [
            {"CPU": max(resources.trainer_num_cpus, 0.001)},
            *[
                {
                    "GPU": 1,
                    "CPU": max(resources.trainer_num_cpus, 0.001),
                }
                for _ in range(distributed.gpus_per_node)
            ],
        ]
        group = placement_group(
            bundles,
            strategy=distributed.placement_strategy,
            name=f"ray-unsloth-ddp-{session_id[-8:]}",
        )
        self._ray.get(group.ready(), timeout=120)
        return group

    def create_training_actor(
        self,
        *,
        base_model: str | None = None,
        lora_rank: int | None = None,
        seed: int | None = None,
        target_modules: list[str] | None = None,
        model_path: str | None = None,
        with_optimizer: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, Any]:
        session_id = f"train-{uuid.uuid4().hex}"
        model_config = self._model_config(base_model)
        lora_config = self._lora_config(
            lora_rank,
            base_model=base_model,
            seed=seed,
            target_modules=target_modules,
        )
        if self.config.distributed.enabled:
            actor = self._create_distributed_training_actor(
                session_id=session_id,
                model_config=model_config,
                lora_config=lora_config,
                model_path=model_path,
                with_optimizer=with_optimizer,
                metadata=metadata or {},
            )
            self.training_actors[session_id] = actor
            return session_id, actor

        options = {
            "num_gpus": self.config.resources.trainer_num_gpus,
            "num_cpus": self.config.resources.trainer_num_cpus,
        }
        strategy = self._strategy(0)
        if strategy is not None:
            options["scheduling_strategy"] = strategy
        actor = TrainerActor.options(**options).remote(
            session_id=session_id,
            model_config=model_config,
            lora_config=lora_config,
            checkpoint_root=self.config.checkpoint_root,
            model_path=model_path,
            with_optimizer=with_optimizer,
            metadata=metadata or {},
        )
        self.training_actors[session_id] = actor
        return session_id, actor

    def _create_distributed_training_actor(
        self,
        *,
        session_id: str,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        model_path: str | None,
        with_optimizer: bool,
        metadata: dict[str, Any],
    ) -> Any:
        distributed = self.config.distributed
        group = self._create_distributed_placement_group(session_id)
        workers = []
        for rank in range(distributed.gpus_per_node):
            options = {
                "num_gpus": 1,
                "num_cpus": self.config.resources.trainer_num_cpus,
                "scheduling_strategy": self._strategy(rank + 1, group),
            }
            workers.append(
                DistributedTrainerWorkerActor.options(**options).remote(
                    session_id=session_id,
                    model_config=model_config,
                    lora_config=lora_config,
                    checkpoint_root=self.config.checkpoint_root,
                    rank=rank,
                    world_size=distributed.gpus_per_node,
                    backend=distributed.backend,
                    model_path=model_path,
                    with_optimizer=with_optimizer,
                    metadata=metadata,
                )
            )
        init_method = self._ray.get(workers[0].get_process_group_endpoint.remote())
        self._ray.get([worker.initialize.remote(init_method) for worker in workers])
        coordinator_options = {
            "num_gpus": 0,
            "num_cpus": self.config.resources.trainer_num_cpus,
            "scheduling_strategy": self._strategy(0, group),
        }
        return DistributedTrainerCoordinatorActor.options(**coordinator_options).remote(
            workers=workers,
            world_size=distributed.gpus_per_node,
        )

    def create_sampler_actors(
        self,
        *,
        base_model: str | None = None,
        model_path: str | None = None,
        replicas: int | None = None,
    ) -> tuple[str, list[Any]]:
        session_id = f"sample-{uuid.uuid4().hex}"
        model_config = self._model_config(base_model)
        lora_config = self._lora_config(None, base_model=base_model)
        replica_count = replicas if replicas is not None else self.config.resources.sampler_replicas
        actors = []
        for index in range(replica_count):
            options = {
                "num_gpus": self.config.resources.sampler_num_gpus,
                "num_cpus": self.config.resources.sampler_num_cpus,
            }
            strategy = self._strategy(index + 1)
            if strategy is not None:
                options["scheduling_strategy"] = strategy
            actors.append(
                SamplerActor.options(**options).remote(
                    session_id=session_id,
                    model_config=model_config,
                    lora_config=lora_config,
                    checkpoint_root=self.config.checkpoint_root,
                    model_path=model_path,
                )
            )
        self.sampler_actors[session_id] = actors
        return session_id, actors

    def _model_config(self, base_model: str | None) -> ModelConfig:
        model_config, _lora_config = self.config.resolve_model_configs(base_model)
        return model_config

    def _lora_config(
        self,
        rank: int | None,
        *,
        base_model: str | None = None,
        seed: int | None = None,
        target_modules: list[str] | None = None,
    ) -> LoRAConfig:
        _model_config, lora_config = self.config.resolve_model_configs(base_model)
        updates: dict[str, Any] = {}
        if rank is not None:
            updates["rank"] = rank
        if seed is not None:
            updates["random_state"] = seed
        if target_modules is not None:
            updates["target_modules"] = target_modules
        if not updates:
            return lora_config
        return replace(lora_config, **updates)
