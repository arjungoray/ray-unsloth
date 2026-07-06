"""Modal runtime provider — optional, lazily imported."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ray_unsloth.providers.base import (
    GPU_CATALOG,
    LaunchPlan,
    ProviderCapabilities,
    RuntimeProvider,
    SessionProtocol,
    ValidationIssue,
    estimate_gpu_fit,
)

if TYPE_CHECKING:
    from ray_unsloth.config import RuntimeConfig

_MODAL_GPUS = ["T4", "L4", "A10G", "L40S", "A100-40GB", "A100-80GB", "H100", "H200", "B200"]


class ModalProvider(RuntimeProvider):
    name = "modal"
    description = "Execute trainer/sampler actors in Modal GPU containers with a local Ray control plane."

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            description=self.description,
            kind="execution",
            multi_node=False,
            live_policy_sampling=True,
            gpu_types=list(_MODAL_GPUS),
            cost_estimation=True,
            requires_packages=["modal"],
            docs_url="https://arjungoray.github.io/ray-unsloth/guides/runtimes",
        )

    def validate(self, config: "RuntimeConfig") -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        modal_config = config.modal
        if modal_config.gpu not in _MODAL_GPUS:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path="modal.gpu",
                    message=f"GPU '{modal_config.gpu}' is not in the known Modal GPU list.",
                    hint=f"Known: {', '.join(_MODAL_GPUS)}.",
                )
            )
        if modal_config.timeout < 300:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path="modal.timeout",
                    message=f"timeout of {modal_config.timeout}s is short for model loading + training.",
                    hint="Model download + 4-bit load alone can take several minutes on cold start.",
                )
            )
        if config.distributed.enabled:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path="distributed.enabled",
                    message="DDP is not supported under the Modal provider (single-container actors).",
                )
            )
        return issues

    def plan(self, config: "RuntimeConfig") -> LaunchPlan:
        modal_config = config.modal
        fit = estimate_gpu_fit(config.model, config.lora, gpu=modal_config.gpu)
        catalog = GPU_CATALOG.get(modal_config.gpu)
        hourly = catalog["hourly_usd"] if catalog else None
        steps = [
            f"Build Modal image (python {modal_config.python_version}, torch/unsloth stack)",
            f"Deploy app '{modal_config.app_name}' with GPU {modal_config.gpu}, "
            f"timeout {modal_config.timeout}s, scaledown {modal_config.scaledown_window}s",
            f"Mount volume '{modal_config.volume_name}' at {modal_config.volume_mount_path}",
            "Start local Ray control plane (namespace ray-unsloth-modal)",
            f"Run {config.model.base_model} (LoRA rank {config.lora.rank}) inside Modal containers",
        ]
        return LaunchPlan(
            provider=self.name,
            summary=f"Run {config.model.base_model} on Modal {modal_config.gpu} GPUs.",
            steps=steps,
            fit=fit,
            estimated_hourly_cost_usd=hourly,
        )

    def connect(self, config: "RuntimeConfig") -> SessionProtocol:
        try:
            import modal  # noqa: F401
        except ImportError:
            raise self._not_available(
                reason="the 'modal' package is not installed",
                hint="pip install 'ray-unsloth[modal]' && modal setup",
            ) from None
        from ray_unsloth.runtime.modal import ModalSession

        return ModalSession(config)
