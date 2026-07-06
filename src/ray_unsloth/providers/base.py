"""Runtime provider abstraction.

A :class:`RuntimeProvider` owns the question "where do trainer and sampler
actors run?". The client layer never talks to Ray, Modal, or a cluster
directly — it asks the provider for a *session* (see :class:`SessionProtocol`)
and drives duck-typed actor handles through ``ray_unsloth.clients._remote``.

Providers come in two kinds:

- **execution** providers can actually create sessions (`local-ray`, `modal`,
  `fake`).
- **planned** providers implement capability discovery, config validation, and
  :meth:`RuntimeProvider.plan` — rendering the launch artifacts you would use
  (a SkyPilot task YAML, a KubeRay ``RayJob``, an ``sbatch`` script) — but
  raise :class:`ProviderNotAvailableError` from :meth:`connect`.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ray_unsloth.errors import ProviderNotAvailableError

if TYPE_CHECKING:
    from ray_unsloth.config import LoRAConfig, ModelConfig, RuntimeConfig


# ---------------------------------------------------------------------------
# Session protocol — the informal contract RaySession/ModalSession already
# satisfy, made explicit.
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionProtocol(Protocol):
    """What a provider's session must offer to back the client layer."""

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
    ) -> tuple[str, Any]: ...

    def create_sampler_actors(
        self,
        *,
        base_model: str | None = None,
        model_path: str | None = None,
        replicas: int | None = None,
    ) -> tuple[str, list[Any]]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Capability / validation / planning types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProviderCapabilities:
    name: str
    description: str
    kind: str  # "execution" | "planned"
    multi_node: bool = False
    live_policy_sampling: bool = True
    gpu_types: list[str] = field(default_factory=list)
    cost_estimation: bool = False
    requires_packages: list[str] = field(default_factory=list)
    requires_commands: list[str] = field(default_factory=list)
    docs_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "multi_node": self.multi_node,
            "live_policy_sampling": self.live_policy_sampling,
            "gpu_types": self.gpu_types,
            "cost_estimation": self.cost_estimation,
            "requires_packages": self.requires_packages,
            "requires_commands": self.requires_commands,
            "docs_url": self.docs_url,
        }


@dataclass(slots=True)
class ValidationIssue:
    severity: str  # "error" | "warning" | "info"
    path: str  # config path, e.g. "modal.gpu"
    message: str
    hint: str | None = None

    def __str__(self) -> str:
        text = f"[{self.severity}] {self.path}: {self.message}"
        if self.hint:
            text += f" ({self.hint})"
        return text


@dataclass(slots=True)
class HealthStatus:
    ok: bool
    detail: str
    checks: dict[str, bool] = field(default_factory=dict)


@dataclass(slots=True)
class LaunchPlan:
    """A dry-run description of what launching under a provider would do."""

    provider: str
    summary: str
    steps: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)  # filename -> content
    fit: "GpuFitReport | None" = None
    estimated_hourly_cost_usd: float | None = None

    def render(self) -> str:
        lines = [f"Launch plan — provider: {self.provider}", self.summary, ""]
        for index, step in enumerate(self.steps, start=1):
            lines.append(f"  {index}. {step}")
        if self.fit is not None:
            lines.append("")
            lines.append(self.fit.render())
        if self.estimated_hourly_cost_usd is not None:
            lines.append("")
            lines.append(f"Estimated cost: ~${self.estimated_hourly_cost_usd:.2f}/hour")
        if self.artifacts:
            lines.append("")
            lines.append("Artifacts:")
            for filename in sorted(self.artifacts):
                lines.append(f"  - {filename}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPU catalog and fit estimation
# ---------------------------------------------------------------------------

# memory_gb, on-demand hourly price (rough public cloud median, mid-2026).
GPU_CATALOG: dict[str, dict[str, float]] = {
    "T4": {"memory_gb": 16, "hourly_usd": 0.35},
    "L4": {"memory_gb": 24, "hourly_usd": 0.70},
    "L40S": {"memory_gb": 48, "hourly_usd": 1.90},
    "A10G": {"memory_gb": 24, "hourly_usd": 1.05},
    "A100-40GB": {"memory_gb": 40, "hourly_usd": 2.80},
    "A100-80GB": {"memory_gb": 80, "hourly_usd": 3.70},
    "H100": {"memory_gb": 80, "hourly_usd": 5.50},
    "H200": {"memory_gb": 141, "hourly_usd": 7.50},
    "B200": {"memory_gb": 192, "hourly_usd": 11.00},
}

_PARAM_COUNT_PATTERN = re.compile(r"(\d+(?:[._]\d+)?)\s*[bB](?![a-zA-Z])")


def parse_param_count(model_name: str) -> float | None:
    """Best-effort parameter count (in billions) parsed from a model name.

    ``"Qwen/Qwen3.5-4B-Instruct"`` -> 4.0, ``"LFM2.5-1.2B"`` -> 1.2. Returns
    ``None`` when no ``<n>B`` token is present.
    """
    matches = _PARAM_COUNT_PATTERN.findall(model_name.replace("_", "."))
    if not matches:
        return None
    # The last match is usually the size token (version numbers come first).
    return float(matches[-1].replace("_", "."))


@dataclass(slots=True)
class GpuFitReport:
    model: str
    gpu: str
    gpu_memory_gb: float
    estimated_required_gb: float | None
    fits: bool | None
    breakdown: dict[str, float] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)

    def render(self) -> str:
        if self.estimated_required_gb is None:
            return (
                f"GPU fit: could not estimate memory for '{self.model}' "
                "(no parameter-count token in the model name)."
            )
        verdict = "fits" if self.fits else "DOES NOT FIT"
        lines = [
            f"GPU fit: {self.model} on {self.gpu} ({self.gpu_memory_gb:.0f} GB): "
            f"~{self.estimated_required_gb:.1f} GB required — {verdict}",
        ]
        for key, value in self.breakdown.items():
            lines.append(f"    {key}: {value:.1f} GB")
        for note in self.assumptions:
            lines.append(f"    note: {note}")
        return "\n".join(lines)


def estimate_gpu_fit(
    model_config: "ModelConfig",
    lora_config: "LoRAConfig",
    *,
    gpu: str,
) -> GpuFitReport:
    """Heuristic memory-fit check for LoRA fine-tuning on a single GPU.

    Rough model: weights (4-bit ≈ 0.55 bytes/param incl. quantization
    overhead, bf16 ≈ 2 bytes/param), LoRA adapters + Adam states, activations
    scaled by sequence length, and a fixed CUDA/runtime floor. Deliberately
    conservative and clearly labeled as an estimate.
    """
    catalog = GPU_CATALOG.get(gpu, GPU_CATALOG.get(gpu.upper(), None))
    memory_gb = catalog["memory_gb"] if catalog else 0.0
    params_b = parse_param_count(model_config.base_model)
    if params_b is None:
        return GpuFitReport(
            model=model_config.base_model,
            gpu=gpu,
            gpu_memory_gb=memory_gb,
            estimated_required_gb=None,
            fits=None,
        )

    bytes_per_param = 0.55 if model_config.load_in_4bit else 2.0
    weights_gb = params_b * bytes_per_param
    # LoRA params scale with rank; Adam keeps 2 fp32 states + fp32 grads.
    lora_gb = params_b * lora_config.rank * 0.004
    activations_gb = params_b * (model_config.max_seq_length / 2048) * 0.35
    runtime_floor_gb = 1.5
    total = weights_gb + lora_gb + activations_gb + runtime_floor_gb

    assumptions = [
        f"{'4-bit quantized' if model_config.load_in_4bit else 'bf16'} base weights",
        f"LoRA rank {lora_config.rank}, max_seq_length {model_config.max_seq_length}",
        "heuristic estimate — verify with a smoke run",
    ]
    if catalog is None:
        assumptions.append(f"unknown GPU '{gpu}' — not in catalog, fit unknown")

    return GpuFitReport(
        model=model_config.base_model,
        gpu=gpu,
        gpu_memory_gb=memory_gb,
        estimated_required_gb=total,
        fits=(total <= memory_gb * 0.95) if catalog else None,
        breakdown={
            "base weights": weights_gb,
            "lora + optimizer": lora_gb,
            "activations": activations_gb,
            "runtime floor": runtime_floor_gb,
        },
        assumptions=assumptions,
    )


# ---------------------------------------------------------------------------
# Provider base class
# ---------------------------------------------------------------------------


class RuntimeProvider(ABC):
    """Base class for runtime providers. Subclasses set ``name``/``description``."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...

    def validate(self, config: "RuntimeConfig") -> list[ValidationIssue]:
        """Provider-specific config validation. Default: no issues."""
        del config
        return []

    @abstractmethod
    def plan(self, config: "RuntimeConfig") -> LaunchPlan: ...

    @abstractmethod
    def connect(self, config: "RuntimeConfig") -> SessionProtocol:
        """Create a live session. Planned providers raise ProviderNotAvailableError."""

    def health(self, config: "RuntimeConfig") -> HealthStatus:
        """Environment readiness without side effects. Default: dependency checks."""
        del config
        capabilities = self.capabilities()
        checks: dict[str, bool] = {}
        for package in capabilities.requires_packages:
            checks[f"package:{package}"] = importlib.util.find_spec(package) is not None
        for command in capabilities.requires_commands:
            checks[f"command:{command}"] = shutil.which(command) is not None
        ok = all(checks.values()) if checks else True
        missing = sorted(name for name, passed in checks.items() if not passed)
        detail = "ready" if ok else f"missing: {', '.join(missing)}"
        return HealthStatus(ok=ok, detail=detail, checks=checks)

    def shutdown(self) -> None:
        """Release any provider-level resources. Sessions are closed separately."""

    # Convenience shared by planned providers.
    def _not_available(self, *, reason: str, hint: str) -> ProviderNotAvailableError:
        return ProviderNotAvailableError(
            f"Provider '{self.name}' cannot execute sessions in this build: {reason}",
            hint=hint,
        )
