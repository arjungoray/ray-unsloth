"""Run two concurrent LFM2.5 1.2B-Instruct LoRA SFT jobs on one shared L4.

This example exercises the multi-tenant backend by creating two independent
LoRA training clients for the same base model at the same time. Each tenant
gets its own adapter state, optimizer state, checkpoint namespace, and W&B run,
while both trainer sessions route to one Modal L4 container pool.

Run:

    python examples/lfm2_5_1_2b_multitenant_sft.py \
        --config configs/lfm2_5_1_2b_1x_l4_multitenant.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ray_unsloth import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient
from ray_unsloth.clients._remote import resolve
from ray_unsloth.clients.sampling import SamplingClient


_SFT_PATH = Path(__file__).with_name("tinker_first_sft_training.py")
_SFT_SPEC = importlib.util.spec_from_file_location("tinker_first_sft_training_recipe", _SFT_PATH)
if _SFT_SPEC is None or _SFT_SPEC.loader is None:
    raise ImportError(f"Could not load SFT helpers from {_SFT_PATH}")
sft = importlib.util.module_from_spec(_SFT_SPEC)
sys.modules[_SFT_SPEC.name] = sft
_SFT_SPEC.loader.exec_module(sft)


BASE_MODEL = "lfm2.5-1.2b-instruct"
SETTINGS_KEY = "lfm2_5_1_2b_multitenant_sft"


@dataclass(frozen=True)
class TenantSpec:
    name: str
    seed: int
    system_prompt: str
    conversations: list[list[dict[str, str]]]
    test_questions: list[str]


@dataclass(frozen=True)
class TenantResult:
    name: str
    session_id: str
    sampler_path: str
    final_loss: float
    elapsed: float


class WandbRunLogger:
    def __init__(self, *, enabled: bool, run: Any | None = None, wandb: Any | None = None) -> None:
        self.enabled = enabled
        self.run = run
        self.wandb = wandb
        self.event_index = 0

    @classmethod
    def start(
        cls,
        *,
        settings: dict[str, Any],
        config_path: str,
        run_name: str,
        tags: list[str],
        run_config: dict[str, Any],
        group: str,
        job_type: str,
    ) -> "WandbRunLogger":
        wandb_settings = dict(settings.get("wandb", {}))
        enabled = bool(wandb_settings.get("enabled", True))
        if not enabled:
            return cls(enabled=False)
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B logging is enabled. Install it with `pip install wandb`.") from exc

        run = wandb.init(
            project=str(wandb_settings.get("project", "ray-unsloth-multitenant")),
            entity=wandb_settings.get("entity"),
            name=run_name,
            group=group,
            job_type=job_type,
            tags=tags,
            notes=wandb_settings.get("notes"),
            mode=wandb_settings.get("mode"),
            reinit="create_new",
            config={
                "config_path": config_path,
                **run_config,
                "wandb": wandb_settings,
            },
        )
        logger = cls(enabled=True, run=run, wandb=wandb)
        logger.define_metrics()
        return logger

    def define_metrics(self) -> None:
        if not self.enabled or self.run is None:
            return
        define_metric = getattr(self.run, "define_metric", None)
        if not callable(define_metric) and self.wandb is not None:
            define_metric = getattr(self.wandb, "define_metric", None)
        if not callable(define_metric):
            return
        define_metric("train/step")
        define_metric("wandb/event_index")
        for prefix in ("progress", "train", "sampling", "tokens", "timing", "tenant", "aggregate"):
            define_metric(f"{prefix}/*", step_metric="train/step")

    def log(self, payload: dict[str, Any], *, step: int) -> None:
        if not self.enabled or self.run is None:
            return
        self.event_index += 1
        self.run.log(
            {"wandb/event_index": self.event_index, "train/step": step, **payload},
            step=self.event_index,
        )

    def table(self, columns: list[str], rows: list[list[Any]]):
        if not self.enabled or self.wandb is None:
            return None
        return self.wandb.Table(columns=columns, data=rows)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()


def load_local_settings(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return dict(data.get("examples", {}).get(SETTINGS_KEY, {}))


def conversation(system_prompt: str, user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def tenant_specs() -> list[TenantSpec]:
    math_prompt = (
        "You are Arc, a compact math tutor. Show the arithmetic in one short line, "
        "then give the final answer."
    )
    style_prompt = (
        "You are Vale, a careful writing coach. Rewrite text with clean structure, "
        "plain language, and no extra flourish."
    )
    return [
        TenantSpec(
            name="arc-math-tutor",
            seed=101,
            system_prompt=math_prompt,
            conversations=[
                conversation(math_prompt, "What is 18 * 7?", "18 * 7 = 126. Final answer: 126."),
                conversation(math_prompt, "Compute 144 / 12.", "144 / 12 = 12. Final answer: 12."),
                conversation(math_prompt, "What is 39 + 58?", "39 + 58 = 97. Final answer: 97."),
                conversation(math_prompt, "Compute 15% of 240.", "0.15 * 240 = 36. Final answer: 36."),
            ],
            test_questions=[
                "What is 27 * 6?",
                "Compute 25% of 88.",
            ],
        ),
        TenantSpec(
            name="vale-writing-coach",
            seed=202,
            system_prompt=style_prompt,
            conversations=[
                conversation(
                    style_prompt,
                    "Rewrite: We are presently endeavoring to improve the system.",
                    "We are improving the system.",
                ),
                conversation(
                    style_prompt,
                    "Rewrite: Due to the fact that logs were missing, debugging was hard.",
                    "Because logs were missing, debugging was hard.",
                ),
                conversation(
                    style_prompt,
                    "Rewrite: Please be advised that the meeting starts at noon.",
                    "The meeting starts at noon.",
                ),
                conversation(
                    style_prompt,
                    "Rewrite: This feature gives users the ability to export files.",
                    "This feature lets users export files.",
                ),
            ],
            test_questions=[
                "Rewrite: At this point in time, the results are not available.",
                "Rewrite: We made a decision to reduce latency.",
            ],
        ),
    ]


def build_prompt(tokenizer: Any, spec: TenantSpec, question: str) -> ModelInput:
    messages = [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": question},
    ]
    tokens = sft.strip_trailing_eos(
        sft.encode_chat(tokenizer, messages, add_generation_prompt=True),
        tokenizer,
    )
    return ModelInput.from_ints(tokens)


def token_count(data: list[Datum]) -> int:
    return sum(len(datum.model_input.to_ints()) for datum in data)


async def train_one_tenant(
    *,
    service_client: ServiceClient,
    spec: TenantSpec,
    settings: dict[str, Any],
    config_path: str,
    group_name: str,
) -> TenantResult:
    steps = int(settings.get("steps", 12))
    learning_rate = float(settings.get("learning_rate", 0.0002))
    max_length = int(settings.get("max_length", 512))
    sample_tokens = int(settings.get("sample_tokens", 128))
    temperature = float(settings.get("temperature", 0.7))
    rank = int(settings.get("rank", 16))
    run_name = f"{settings.get('wandb', {}).get('name', 'lfm2.5-1.2b-instruct-1xl4-multitenant-sft')}/{spec.name}"
    logger = WandbRunLogger.start(
        settings=settings,
        config_path=config_path,
        run_name=run_name,
        tags=[*list(settings.get("wandb", {}).get("tags", [])), spec.name],
        group=group_name,
        job_type="tenant-train",
        run_config={
            "tenant": spec.name,
            "base_model": BASE_MODEL,
            "rank": rank,
            "seed": spec.seed,
            "steps": steps,
            "learning_rate": learning_rate,
            "max_length": max_length,
        },
    )
    started = time.time()
    try:
        logger.log({"progress/phase": "model_create_started", "tenant/name": spec.name}, step=0)
        training_client = await service_client.create_lora_training_client_async(
            base_model=BASE_MODEL,
            rank=rank,
            seed=spec.seed,
            user_metadata={"tenant": spec.name, "example": SETTINGS_KEY},
        )
        tokenizer = resolve(training_client.get_tokenizer())
        training_data = [
            sft.conversation_to_datum(item, tokenizer, max_length=max_length)
            for item in spec.conversations
        ]
        logger.log(
            {
                "progress/phase": "model_ready",
                "tenant/session_id": training_client.session_id,
                "data/example_count": len(training_data),
                "tokens/train_batch": token_count(training_data),
            },
            step=0,
        )

        final_loss = 0.0
        for step in range(steps):
            t0 = time.time()
            logger.log({"progress/phase": "train_step_started"}, step=step)
            fwdbwd_future = await training_client.forward_backward_async(training_data, "cross_entropy")
            optim_future = await training_client.optim_step_async(AdamParams(learning_rate=learning_rate))
            fwdbwd_result = await fwdbwd_future.result_async()
            optim_result = await optim_future.result_async()
            final_loss = sft.weighted_mean_nll(training_data, fwdbwd_result.loss_fn_outputs)
            logger.log(
                {
                    "progress/phase": "train_step_finished",
                    "train/loss": final_loss,
                    "train/optimizer_step": optim_result.step,
                    "timing/step_seconds": time.time() - t0,
                    "tokens/train_total": token_count(training_data) * (step + 1),
                },
                step=step,
            )

        sampler_name = f"{spec.name}-lfm2.5-1.2b-instruct-sft"
        saved = await training_client.save_weights_for_sampler_async(name=sampler_name).result_async()
        sampling_client = SamplingClient(
            session_id=f"{training_client.session_id}-live-sampler",
            actors=[training_client._actor],
        )
        params = SamplingParams(max_tokens=sample_tokens, temperature=temperature)
        rows = []
        sampled_tokens = 0
        for question in spec.test_questions:
            prompt = build_prompt(tokenizer, spec, question)
            sample = await sampling_client.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=params,
            )
            response = await sample.result_async()
            sequence = response.sequences[0]
            answer = sequence.text or sft.decode_tokens(tokenizer, sequence.tokens)
            sampled_tokens += len(sequence.tokens)
            rows.append([spec.name, question, answer])
        table = logger.table(["tenant", "question", "answer"], rows)
        payload: dict[str, Any] = {
            "progress/phase": "sampling_finished",
            "sampling/sample_count": len(rows),
            "tokens/sample_total": sampled_tokens,
            "checkpoint/sampler_path": saved.path,
            "timing/elapsed_seconds": time.time() - started,
        }
        if table is not None:
            payload["sampling/examples"] = table
        logger.log(payload, step=steps)
        return TenantResult(
            name=spec.name,
            session_id=training_client.session_id,
            sampler_path=saved.path,
            final_loss=final_loss,
            elapsed=time.time() - started,
        )
    finally:
        logger.finish()


async def train(args: argparse.Namespace) -> None:
    settings = load_local_settings(args.config)
    group_name = str(settings.get("wandb", {}).get("name", "lfm2.5-1.2b-instruct-1xl4-multitenant-sft"))
    orchestrator = WandbRunLogger.start(
        settings=settings,
        config_path=str(args.config),
        run_name=f"{group_name}/orchestrator",
        tags=[*list(settings.get("wandb", {}).get("tags", [])), "orchestrator"],
        group=group_name,
        job_type="orchestrator",
        run_config={"base_model": BASE_MODEL, "tenant_count": len(tenant_specs())},
    )
    service_client = ServiceClient(config=args.config)
    try:
        capabilities = service_client.get_server_capabilities()
        orchestrator.log(
            {
                "progress/phase": "service_ready",
                "aggregate/tenant_count": len(tenant_specs()),
                "aggregate/max_concurrent_trainers": capabilities.max_concurrent_trainers,
            },
            step=0,
        )
        started = time.time()
        results = await asyncio.gather(
            *[
                train_one_tenant(
                    service_client=service_client,
                    spec=spec,
                    settings=settings,
                    config_path=str(args.config),
                    group_name=group_name,
                )
                for spec in tenant_specs()
            ]
        )
        rows = [
            [result.name, result.session_id, result.final_loss, result.elapsed, result.sampler_path]
            for result in results
        ]
        table = orchestrator.table(
            ["tenant", "session_id", "final_loss", "elapsed_seconds", "sampler_path"],
            rows,
        )
        payload: dict[str, Any] = {
            "progress/phase": "all_tenants_finished",
            "aggregate/tenant_count": len(results),
            "aggregate/elapsed_seconds": time.time() - started,
        }
        if table is not None:
            payload["aggregate/results"] = table
        orchestrator.log(payload, step=int(settings.get("steps", 12)))
        for result in results:
            print(
                f"{result.name}: session={result.session_id} "
                f"loss={result.final_loss:.4f} sampler={result.sampler_path}"
            )
    finally:
        service_client.close()
        orchestrator.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lfm2_5_1_2b_1x_l4_multitenant.yaml")
    args = parser.parse_args()

    asyncio.run(train(args))


if __name__ == "__main__":
    main()
