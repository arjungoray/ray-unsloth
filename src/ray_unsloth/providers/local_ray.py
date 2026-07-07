"""Local Ray runtime provider — the first-class default."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from ray_unsloth.providers.base import (
    LaunchPlan,
    ProviderCapabilities,
    RuntimeProvider,
    SessionProtocol,
    ValidationIssue,
    estimate_gpu_fit,
)

if TYPE_CHECKING:
    from ray_unsloth.config import RuntimeConfig


class LocalRayProvider(RuntimeProvider):
    name = "local-ray"
    description = "Run trainer/sampler actors on a local or attached Ray cluster."

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            description=self.description,
            kind="execution",
            multi_node=False,  # single-node DDP today; multi-node on roadmap
            live_policy_sampling=True,
            gpu_types=["any locally visible GPU"],
            cost_estimation=False,
            requires_packages=["ray"],
            docs_url="https://arjungoray.github.io/ray-unsloth/guides/runtimes",
        )

    def validate(self, config: RuntimeConfig) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if config.modal.enabled:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path="modal.enabled",
                    message="modal.enabled is true but the selected provider is local-ray.",
                    hint="Set provider: modal (or remove modal.enabled) to avoid ambiguity.",
                )
            )
        if config.distributed.enabled and config.distributed.num_nodes > 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path="distributed.num_nodes",
                    message="local-ray currently supports single-node DDP only.",
                )
            )
        resources = config.resources
        if resources.trainer_num_gpus == 0 and not config.modal.enabled:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path="resources.trainer_num_gpus",
                    message="trainer_num_gpus is 0 — training will run on CPU-visible actors only.",
                    hint="This is normal for Modal-backed configs but usually wrong for local-ray.",
                )
            )
        return issues

    def plan(self, config: RuntimeConfig) -> LaunchPlan:
        resources = config.resources
        gpu_hint = "local GPU"
        fit = None
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_hint = torch.cuda.get_device_name(0)
            except Exception:
                pass
        if config.modal.gpu:
            fit = estimate_gpu_fit(config.model, config.lora, gpu=config.modal.gpu)
        steps = [
            f"ray.init(address={config.ray.address!r}, namespace={config.ray.namespace!r})",
            f"Create trainer actor(s): {resources.trainer_replicas} × "
            f"{{GPU: {resources.trainer_num_gpus}, CPU: {resources.trainer_num_cpus}}} "
            f"(placement: {resources.placement_strategy})",
            f"Create sampler actor(s): {resources.sampler_replicas} × "
            f"{{GPU: {resources.sampler_num_gpus}, CPU: {resources.sampler_num_cpus}}}",
            f"Load {config.model.base_model} with LoRA rank {config.lora.rank} on {gpu_hint}",
            f"Checkpoints under {config.checkpoint_root}/",
        ]
        if config.distributed.enabled:
            steps.insert(
                2,
                f"DDP: {config.distributed.gpus_per_node} worker(s), backend {config.distributed.backend}",
            )
        return LaunchPlan(
            provider=self.name,
            summary=f"Run {config.model.base_model} on the local Ray runtime.",
            steps=steps,
            fit=fit,
        )

    def connect(self, config: RuntimeConfig) -> SessionProtocol:
        from ray_unsloth.runtime.ray import RaySession

        return RaySession(config)
