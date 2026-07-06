"""Planned runtime providers.

These providers implement capability discovery, config validation, and
:meth:`plan` — rendering the real launch artifacts you would submit (SkyPilot
task YAML, KubeRay ``RayJob``, ``sbatch`` script, RunPod pod spec) — but do
not yet execute sessions. ``connect()`` raises
:class:`ProviderNotAvailableError` with the artifact-driven path forward:
render the plan, launch a Ray cluster with it, then point ``local-ray`` at
that cluster via ``ray.address``.

That is not a cop-out — it is the actual topology: every one of these
systems provisions machines and starts Ray; the ray-unsloth control plane
then attaches with ``ray.address: "ray://<head>:10001"``. What is "planned"
is the automated provision-attach-teardown lifecycle.
"""

from __future__ import annotations

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

_DOCS = "https://arjungoray.github.io/ray-unsloth/guides/providers"


def _gpu_for(config: "RuntimeConfig") -> str:
    return (config.provider_options or {}).get("gpu", config.modal.gpu or "L4")


def _attach_steps(head_address: str) -> list[str]:
    return [
        f"Attach ray-unsloth: set `ray.address: \"{head_address}\"` and `provider: local-ray` in your config",
        "Run your training loop as usual — actors schedule onto the remote cluster",
    ]


class _PlannedProvider(RuntimeProvider):
    kind = "planned"

    def connect(self, config: "RuntimeConfig") -> SessionProtocol:
        del config
        raise self._not_available(
            reason="automated provisioning for this provider is on the roadmap",
            hint=(
                f"Run `ray-unsloth plan --provider {self.name}` to render launch artifacts, "
                f"start the cluster with them, then set provider: local-ray with "
                f"ray.address pointing at the cluster head. Docs: {_DOCS}"
            ),
        )


class KubeRayProvider(_PlannedProvider):
    name = "kuberay"
    description = "Kubernetes clusters via the KubeRay operator (RayJob/RayCluster CRs)."

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            description=self.description,
            kind="planned",
            multi_node=True,
            gpu_types=["any GPU your cluster nodes expose"],
            requires_commands=["kubectl"],
            docs_url=_DOCS,
        )

    def plan(self, config: "RuntimeConfig") -> LaunchPlan:
        options = config.provider_options or {}
        gpu = _gpu_for(config)
        namespace = options.get("namespace", "default")
        image = options.get("image", "rayproject/ray:2.55.1-py311-gpu")
        workers = int(options.get("workers", max(1, config.resources.trainer_replicas)))
        cluster_yaml = f"""apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ray-unsloth
  namespace: {namespace}
spec:
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
    template:
      spec:
        containers:
          - name: ray-head
            image: {image}
            resources:
              limits: {{cpu: "4", memory: 16Gi}}
  workerGroupSpecs:
    - groupName: gpu-workers
      replicas: {workers}
      minReplicas: 0
      maxReplicas: {workers}
      rayStartParams: {{}}
      template:
        spec:
          containers:
            - name: ray-worker
              image: {image}
              resources:
                limits:
                  cpu: "8"
                  memory: 32Gi
                  nvidia.com/gpu: "1"
"""
        return LaunchPlan(
            provider=self.name,
            summary=f"Provision a RayCluster with {workers} GPU worker(s) via KubeRay, then attach local-ray.",
            steps=[
                "Install the KubeRay operator (helm install kuberay-operator kuberay/kuberay-operator)",
                f"kubectl apply -f ray-cluster.yaml (namespace {namespace})",
                "kubectl port-forward svc/ray-unsloth-head-svc 10001:10001",
                *_attach_steps("ray://127.0.0.1:10001"),
            ],
            artifacts={"ray-cluster.yaml": cluster_yaml},
            fit=estimate_gpu_fit(config.model, config.lora, gpu=gpu),
        )


class SkyPilotProvider(_PlannedProvider):
    name = "skypilot"
    description = "Multi-cloud GPU provisioning with cost-optimized failover via SkyPilot."

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            description=self.description,
            kind="planned",
            multi_node=True,
            gpu_types=["any_of constraint sets across AWS/GCP/Azure/Lambda/RunPod..."],
            cost_estimation=True,
            requires_packages=["sky"],
            requires_commands=["sky"],
            docs_url=_DOCS,
        )

    def validate(self, config: "RuntimeConfig") -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        options = config.provider_options or {}
        if "gpu" not in options and not config.modal.gpu:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path="provider_options.gpu",
                    message="No GPU type specified; the plan defaults to L4.",
                    hint="Set provider_options.gpu (e.g. A100-80GB) for an accurate plan.",
                )
            )
        return issues

    def plan(self, config: "RuntimeConfig") -> LaunchPlan:
        options = config.provider_options or {}
        gpu = _gpu_for(config)
        sky_gpu = gpu.replace("-40GB", ":40GB").replace("-80GB", ":80GB").split(":")[0]
        use_spot = bool(options.get("use_spot", True))
        task_yaml = f"""# sky launch -c ray-unsloth task.yaml
resources:
  any_of:
    - accelerators: {sky_gpu}:1
      use_spot: {str(use_spot).lower()}
    - accelerators: {sky_gpu}:1
      use_spot: false

setup: |
  pip install "ray[default]>=2.55.1" "ray-unsloth[unsloth]"

run: |
  ray start --head --port=6379 --dashboard-host=0.0.0.0
  echo "Head ready at $(hostname -I | awk '{{print $1}}'):10001"
  sleep infinity
"""
        return LaunchPlan(
            provider=self.name,
            summary=f"Provision a {sky_gpu} Ray head via SkyPilot (spot-first failover), then attach local-ray.",
            steps=[
                "pip install 'skypilot[aws,gcp]' && sky check",
                "sky launch -c ray-unsloth task.yaml   # prints ranked cost plan before provisioning",
                "ssh -L 10001:localhost:10001 ray-unsloth   # or use the printed head IP",
                *_attach_steps("ray://127.0.0.1:10001"),
                "sky down ray-unsloth   # teardown when finished",
            ],
            artifacts={"task.yaml": task_yaml},
            fit=estimate_gpu_fit(config.model, config.lora, gpu=gpu),
        )


class SlurmProvider(_PlannedProvider):
    name = "slurm"
    description = "HPC clusters via sbatch scripts that start a Ray head on the allocation."

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            description=self.description,
            kind="planned",
            multi_node=True,
            gpu_types=["cluster-dependent (via --gres)"],
            requires_commands=["sbatch"],
            docs_url=_DOCS,
        )

    def plan(self, config: "RuntimeConfig") -> LaunchPlan:
        options = config.provider_options or {}
        gpu = _gpu_for(config)
        partition = options.get("partition", "gpu")
        time_limit = options.get("time", "04:00:00")
        sbatch = f"""#!/bin/bash
#SBATCH --job-name=ray-unsloth
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={time_limit}

module load cuda 2>/dev/null || true
source "$HOME/ray-unsloth-venv/bin/activate"

ray start --head --port=6379 --dashboard-host=0.0.0.0
echo "Ray head on $(hostname):10001"
sleep infinity
"""
        return LaunchPlan(
            provider=self.name,
            summary=f"Reserve a GPU node on partition '{partition}' and start a Ray head, then attach local-ray.",
            steps=[
                "python -m venv ~/ray-unsloth-venv && pip install 'ray-unsloth[unsloth]'",
                "sbatch ray-head.sbatch",
                "squeue --me   # find the allocated node",
                "ssh -L 10001:<node>:10001 <login-node>",
                *_attach_steps("ray://127.0.0.1:10001"),
            ],
            artifacts={"ray-head.sbatch": sbatch},
            fit=estimate_gpu_fit(config.model, config.lora, gpu=gpu),
        )


class RunPodProvider(_PlannedProvider):
    name = "runpod"
    description = "BYOC GPU marketplaces (RunPod/Lambda/CoreWeave-style): rent a pod, start Ray, attach."

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            description=self.description,
            kind="planned",
            multi_node=False,
            gpu_types=["marketplace inventory (L4/A100/H100/H200/B200...)"],
            cost_estimation=True,
            requires_packages=["runpod"],
            docs_url=_DOCS,
        )

    def plan(self, config: "RuntimeConfig") -> LaunchPlan:
        gpu = _gpu_for(config)
        fit = estimate_gpu_fit(config.model, config.lora, gpu=gpu)
        bootstrap = """#!/bin/bash
# Run inside the rented pod (any provider with SSH + CUDA image)
pip install "ray[default]>=2.55.1" "ray-unsloth[unsloth]"
ray start --head --port=6379 --dashboard-host=0.0.0.0
echo "Ray head ready on port 10001"
"""
        return LaunchPlan(
            provider=self.name,
            summary=f"Rent a {gpu} pod, bootstrap a Ray head, then attach local-ray over an SSH tunnel.",
            steps=[
                f"Rent a {gpu} pod with a CUDA 12 base image and SSH access",
                "Copy + run bootstrap.sh inside the pod",
                "ssh -L 10001:localhost:10001 root@<pod-ip> -p <ssh-port>",
                *_attach_steps("ray://127.0.0.1:10001"),
                "Stop the pod when finished (billing is per-minute)",
            ],
            artifacts={"bootstrap.sh": bootstrap},
            fit=fit,
        )
