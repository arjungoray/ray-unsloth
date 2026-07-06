"""Runtime providers and provider resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ray_unsloth.errors import PluginError, ProviderError
from ray_unsloth.plugins import providers as provider_registry
from ray_unsloth.providers.base import (
    GPU_CATALOG,
    GpuFitReport,
    HealthStatus,
    LaunchPlan,
    ProviderCapabilities,
    RuntimeProvider,
    SessionProtocol,
    ValidationIssue,
    estimate_gpu_fit,
    parse_param_count,
)

if TYPE_CHECKING:
    from ray_unsloth.config import RuntimeConfig

__all__ = [
    "GPU_CATALOG",
    "GpuFitReport",
    "HealthStatus",
    "LaunchPlan",
    "ProviderCapabilities",
    "RuntimeProvider",
    "SessionProtocol",
    "ValidationIssue",
    "estimate_gpu_fit",
    "get_provider",
    "list_providers",
    "parse_param_count",
    "resolve_provider",
    "resolve_provider_name",
]

# Built-ins register lazily: nothing (Ray, Modal, torch) is imported until a
# provider is actually resolved.
_BUILTINS = {
    "local-ray": ("ray_unsloth.providers.local_ray:LocalRayProvider", "Local or attached Ray cluster (default)"),
    "modal": ("ray_unsloth.providers.modal_provider:ModalProvider", "Modal GPU containers (optional extra)"),
    "fake": ("ray_unsloth.providers.fake:FakeProvider", "GPU-free in-process engine for tests/demos"),
    "kuberay": ("ray_unsloth.providers.planned:KubeRayProvider", "Kubernetes via KubeRay (plan-only)"),
    "skypilot": ("ray_unsloth.providers.planned:SkyPilotProvider", "Multi-cloud via SkyPilot (plan-only)"),
    "slurm": ("ray_unsloth.providers.planned:SlurmProvider", "HPC via Slurm sbatch (plan-only)"),
    "runpod": ("ray_unsloth.providers.planned:RunPodProvider", "BYOC GPU marketplaces (plan-only)"),
}

for _name, (_target, _description) in _BUILTINS.items():
    if _name not in provider_registry:
        provider_registry.register_lazy(_name, _target, description=_description)


def get_provider(name: str) -> RuntimeProvider:
    """Instantiate the provider registered under ``name``."""
    try:
        provider_cls = provider_registry.get(name)
    except PluginError as exc:
        raise ProviderError(str(exc)) from None
    provider = provider_cls() if isinstance(provider_cls, type) else provider_cls
    if not isinstance(provider, RuntimeProvider):
        raise ProviderError(f"Registered provider '{name}' is not a RuntimeProvider (got {type(provider).__name__}).")
    return provider


def resolve_provider_name(config: RuntimeConfig) -> str:
    """The provider a config selects, honoring the legacy ``modal.enabled`` switch."""
    if config.provider:
        return config.provider
    if config.modal.enabled:
        return "modal"
    return "local-ray"


def resolve_provider(config: RuntimeConfig) -> RuntimeProvider:
    return get_provider(resolve_provider_name(config))


def list_providers() -> list[str]:
    """Registered provider names without instantiating anything."""
    return provider_registry.names()
