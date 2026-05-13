"""Run three concurrent LFM2.5 1.2B-Instruct LoRA RL jobs on one shared L4.

All tenants train with the same low-level RL primitive shape:

1. sample grouped completions from the live policy actor
2. grade each completion with a verifiable math reward
3. build policy datums with old logprobs and advantages
4. update with ``loss_fn="importance_sampling"``

Run:

    python examples/lfm2_5_1_2b_multitenant_rl.py \
        --config configs/lfm2_5_1_2b_1x_l4_multitenant_rl.yaml
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

from ray_unsloth import AdamParams, SamplingParams, ServiceClient
from ray_unsloth.clients._remote import resolve


_RL_PATH = Path(__file__).with_name("qwen3_5_9b_rl_training.py")
_RL_SPEC = importlib.util.spec_from_file_location("qwen3_5_9b_rl_training_recipe", _RL_PATH)
if _RL_SPEC is None or _RL_SPEC.loader is None:
    raise ImportError(f"Could not load RL helpers from {_RL_PATH}")
rl = importlib.util.module_from_spec(_RL_SPEC)
sys.modules[_RL_SPEC.name] = rl
_RL_SPEC.loader.exec_module(rl)


BASE_MODEL = "lfm2.5-1.2b-instruct"
SETTINGS_KEY = "lfm2_5_1_2b_multitenant_rl"


@dataclass(frozen=True)
class TenantSpec:
    name: str
    seed: int
    problems: list[Any]


@dataclass(frozen=True)
class TenantResult:
    name: str
    session_id: str
    sampler_path: str
    final_reward: float
    final_loss: float
    elapsed: float
    train_tokens: int
    sample_tokens: int
    prefill_tokens: int


class WandbRunLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        run: Any | None = None,
        wandb: Any | None = None,
        max_completion_rows: int = 32,
    ) -> None:
        self.enabled = enabled
        self.run = run
        self.wandb = wandb
        self.max_completion_rows = max_completion_rows
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
            config={"config_path": config_path, **run_config, "wandb": wandb_settings},
        )
        logger = cls(
            enabled=True,
            run=run,
            wandb=wandb,
            max_completion_rows=int(wandb_settings.get("log_completions", 32)),
        )
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
        for prefix in ("progress", "reward", "rollout", "policy", "train", "timing", "tokens", "tenant", "aggregate"):
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
        return self.wandb.Table(columns=columns, data=rows[: self.max_completion_rows])

    def histogram(self, values: list[float]):
        if not self.enabled or self.wandb is None or not values:
            return None
        return self.wandb.Histogram(values)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()


def load_local_settings(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return dict(data.get("examples", {}).get(SETTINGS_KEY, {}))


def tenant_specs() -> list[TenantSpec]:
    return [
        TenantSpec(
            name="arith-rl-tenant",
            seed=301,
            problems=[
                rl.MathProblem("Compute 37 * 24 - 156.", "732"),
                rl.MathProblem("Compute 125% of 64, then subtract 17.", "63"),
                rl.MathProblem("The average of 6 numbers is 18. Five numbers are 11, 14, 20, 21, and 25. What is the sixth?", "17"),
                rl.MathProblem("Solve for x: 5x - 7 = 3x + 29.", "18"),
            ],
        ),
        TenantSpec(
            name="word-problem-rl-tenant",
            seed=302,
            problems=[
                rl.MathProblem("A sequence starts at 7. Each next term is double the previous term minus 3. What is the 5th term?", "67"),
                rl.MathProblem("What is the remainder when 17^3 is divided by 19?", "11"),
                rl.MathProblem("A rectangle has perimeter 90. Its length is 3 more than twice its width. What is its area?", "434"),
                rl.MathProblem("A jar has red and blue marbles in a 3:5 ratio. After adding 8 red marbles, the ratio is 5:7. What was the initial total number of marbles?", "112"),
            ],
        ),
        TenantSpec(
            name="algebra-rl-tenant",
            seed=303,
            problems=[
                rl.MathProblem("Solve for x: 3(x + 4) = 2x + 31.", "19"),
                rl.MathProblem("If y = 2x^2 - 3x + 5, what is y when x = 6?", "59"),
                rl.MathProblem("The sum of three consecutive integers is 84. What is the largest integer?", "29"),
                rl.MathProblem("A line has slope 4 and passes through (2, 9). What is its y-intercept?", "1"),
            ],
        ),
    ]


def rollout_token_count(rollouts: list[Any]) -> int:
    return sum(len(completion.tokens) for rollout in rollouts for completion in rollout.completions)


def rollout_prefill_token_count(rollouts: list[Any]) -> int:
    return sum(int(rollout.prompt.length) for rollout in rollouts)


def train_token_count(datums: list[Any]) -> int:
    return sum(int(datum.model_input.length) for datum in datums)


def rollout_rows(tenant: str, step: int, rollouts: list[Any]) -> list[list[Any]]:
    rows = []
    for row in rl.rollout_flat_rows(rollouts):
        rows.append(
            [
                tenant,
                step,
                row["problem_index"],
                row["completion_index"],
                row["question"],
                row["answer"],
                row["reward"],
                row["advantage"],
                row["tokens"],
                row["degenerate"],
                row["text"],
            ]
        )
    return rows


async def train_one_tenant(
    *,
    service_client: ServiceClient,
    spec: TenantSpec,
    settings: dict[str, Any],
    config_path: str,
    group_name: str,
) -> TenantResult:
    steps = int(settings.get("steps", 8))
    batch_size = int(settings.get("batch_size", 1))
    group_size = int(settings.get("group_size", 3))
    learning_rate = float(settings.get("learning_rate", 0.00004))
    max_tokens = int(settings.get("max_tokens", 96))
    temperature = float(settings.get("temperature", 0.8))
    top_p = float(settings.get("top_p", 0.95))
    exploration_temperature = float(settings.get("exploration_temperature", 1.0))
    exploration_top_p = float(settings.get("exploration_top_p", 0.95))
    min_train_datums = int(settings.get("min_train_datums", 1))
    sft_anchor_weight = float(settings.get("sft_anchor_weight", 0.2))
    rank = int(settings.get("rank", 16))
    run_name = f"{settings.get('wandb', {}).get('name', 'lfm2.5-1.2b-instruct-1xl4-multitenant-rl')}/{spec.name}"
    logger = WandbRunLogger.start(
        settings=settings,
        config_path=config_path,
        run_name=run_name,
        tags=[*list(settings.get("wandb", {}).get("tags", [])), spec.name],
        group=group_name,
        job_type="tenant-rl",
        run_config={
            "tenant": spec.name,
            "base_model": BASE_MODEL,
            "rank": rank,
            "seed": spec.seed,
            "steps": steps,
            "batch_size": batch_size,
            "group_size": group_size,
            "learning_rate": learning_rate,
            "problem_count": len(spec.problems),
        },
    )
    started = time.time()
    final_reward = 0.0
    final_loss = 0.0
    total_train_tokens = 0
    total_sample_tokens = 0
    total_prefill_tokens = 0
    try:
        logger.log({"progress/phase": "model_create_started", "tenant/name": spec.name}, step=0)
        training_client = await service_client.create_lora_training_client_async(
            base_model=BASE_MODEL,
            rank=rank,
            seed=spec.seed,
            user_metadata={"tenant": spec.name, "example": SETTINGS_KEY},
        )
        tokenizer = resolve(training_client.get_tokenizer())
        sampling_client = training_client.create_live_sampling_client(name=f"{spec.name}-live-policy")
        logger.log(
            {"progress/phase": "model_ready", "tenant/session_id": training_client.session_id},
            step=0,
        )

        adam_params = AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95)
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        exploration_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=max(temperature, exploration_temperature),
            top_p=exploration_top_p,
        )

        for step in range(steps):
            t0 = time.time()
            start = (step * batch_size) % len(spec.problems)
            batch = [spec.problems[(start + offset) % len(spec.problems)] for offset in range(batch_size)]
            logger.log(
                {
                    "progress/phase": "sampling_started",
                    "data/problem_count": len(batch),
                    "sampling/group_size": group_size,
                },
                step=step,
            )
            rollouts = await rl.collect_rollouts(
                tokenizer=tokenizer,
                sampling_client=sampling_client,
                problems=batch,
                group_size=group_size,
                sampling_params=sampling_params,
                degenerate_reward_baseline=rl.DEGENERATE_REWARD_BASELINE,
            )
            step_prefill_tokens = rollout_prefill_token_count(rollouts)
            step_sample_tokens = rollout_token_count(rollouts)
            datums = [datum for rollout in rollouts for datum in rl.rollout_to_datums(rollout)]
            used_exploration_retry = False
            if len(datums) < min_train_datums:
                retry_rollouts = await rl.collect_rollouts(
                    tokenizer=tokenizer,
                    sampling_client=sampling_client,
                    problems=batch,
                    group_size=group_size,
                    sampling_params=exploration_params,
                    degenerate_reward_baseline=rl.DEGENERATE_REWARD_BASELINE,
                )
                step_prefill_tokens += rollout_prefill_token_count(retry_rollouts)
                step_sample_tokens += rollout_token_count(retry_rollouts)
                rollouts = retry_rollouts
                datums = [datum for rollout in rollouts for datum in rl.rollout_to_datums(rollout)]
                used_exploration_retry = True

            expert_datums = [
                rl.build_supervised_datum(tokenizer=tokenizer, problem=problem, weight=sft_anchor_weight)
                for problem in batch
                if sft_anchor_weight > 0.0
            ]
            step_train_tokens = train_token_count(datums) + train_token_count(expert_datums)
            if datums or expert_datums:
                rl_future = None
                if datums:
                    rl_future = await training_client.forward_backward_async(
                        datums,
                        loss_fn="importance_sampling",
                    )
                sft_future = None
                if expert_datums:
                    sft_future = await training_client.forward_backward_async(
                        expert_datums,
                        loss_fn="cross_entropy",
                    )
                optim_future = await training_client.optim_step_async(adam_params)
                rl_result = await rl_future.result_async() if rl_future is not None else None
                sft_result = await sft_future.result_async() if sft_future is not None else None
                optim_result = await optim_future.result_async()
                if rl_result is not None:
                    mean_logprob, mean_ratio = rl.policy_loss_summary(rl_result.loss_fn_outputs)
                    rl_loss = float(rl_result.loss)
                else:
                    mean_logprob = 0.0
                    mean_ratio = 0.0
                    rl_loss = 0.0
                sft_loss = float(sft_result.loss) if sft_result is not None else 0.0
                optimizer_step = optim_result.step
            else:
                mean_logprob = 0.0
                mean_ratio = 0.0
                rl_loss = 0.0
                sft_loss = 0.0
                optimizer_step = None

            final_loss = rl_loss + sft_loss
            final_reward = sum(rollout.mean_reward for rollout in rollouts) / len(rollouts)
            total_train_tokens += step_train_tokens
            total_sample_tokens += step_sample_tokens
            total_prefill_tokens += step_prefill_tokens
            rewards = [completion.reward for rollout in rollouts for completion in rollout.completions]
            advantages = [completion.advantage for rollout in rollouts for completion in rollout.completions]
            rows = rollout_rows(spec.name, step, rollouts)
            table = logger.table(
                [
                    "tenant",
                    "step",
                    "problem_index",
                    "completion_index",
                    "question",
                    "answer",
                    "reward",
                    "advantage",
                    "tokens",
                    "degenerate",
                    "text",
                ],
                rows,
            )
            payload: dict[str, Any] = {
                "progress/phase": "train_step_finished",
                "reward/mean": final_reward,
                "reward/min": min(rewards) if rewards else 0.0,
                "reward/max": max(rewards) if rewards else 0.0,
                "reward/histogram": logger.histogram([float(reward) for reward in rewards]),
                "rollout/completion_count": len(rows),
                "rollout/degenerate_count": sum(1 for rollout in rollouts if rollout.degenerate),
                "rollout/used_exploration_retry": 1.0 if used_exploration_retry else 0.0,
                "rollout/advantage_histogram": logger.histogram([float(advantage) for advantage in advantages]),
                "policy/mean_logprob": mean_logprob,
                "policy/mean_ratio": mean_ratio,
                "data/rl_datums": len(datums),
                "data/expert_datums": len(expert_datums),
                "train/loss": final_loss,
                "train/rl_loss": rl_loss,
                "train/sft_anchor_loss": sft_loss,
                "train/optimizer_step": optimizer_step,
                "tokens/prefill_step": step_prefill_tokens,
                "tokens/sample_step": step_sample_tokens,
                "tokens/train_step": step_train_tokens,
                "tokens/prefill_total": total_prefill_tokens,
                "tokens/sample_total": total_sample_tokens,
                "tokens/train_total": total_train_tokens,
                "timing/step_seconds": time.time() - t0,
            }
            if table is not None:
                payload["rollout/completions"] = table
            logger.log(payload, step=step)
            print(
                f"{spec.name} step {step:2d}: reward={final_reward:.3f} "
                f"datums={len(datums)} expert={len(expert_datums)} loss={final_loss:.3f}"
            )

        saved = await training_client.save_weights_for_sampler_async(name=f"{spec.name}-lfm2.5-1.2b-instruct-rl").result_async()
        logger.log(
            {
                "progress/phase": "saved",
                "checkpoint/sampler_path": saved.path,
                "tokens/prefill_total": total_prefill_tokens,
                "tokens/sample_total": total_sample_tokens,
                "tokens/train_total": total_train_tokens,
                "timing/elapsed_seconds": time.time() - started,
            },
            step=steps,
        )
        return TenantResult(
            name=spec.name,
            session_id=training_client.session_id,
            sampler_path=saved.path,
            final_reward=final_reward,
            final_loss=final_loss,
            elapsed=time.time() - started,
            train_tokens=total_train_tokens,
            sample_tokens=total_sample_tokens,
            prefill_tokens=total_prefill_tokens,
        )
    finally:
        logger.finish()


async def train(args: argparse.Namespace) -> None:
    settings = load_local_settings(args.config)
    specs = tenant_specs()
    group_name = str(settings.get("wandb", {}).get("name", "lfm2.5-1.2b-instruct-1xl4-multitenant-rl"))
    orchestrator = WandbRunLogger.start(
        settings=settings,
        config_path=str(args.config),
        run_name=f"{group_name}/orchestrator",
        tags=[*list(settings.get("wandb", {}).get("tags", [])), "orchestrator"],
        group=group_name,
        job_type="orchestrator",
        run_config={"base_model": BASE_MODEL, "tenant_count": len(specs)},
    )
    service_client = ServiceClient(config=args.config)
    try:
        capabilities = service_client.get_server_capabilities()
        orchestrator.log(
            {
                "progress/phase": "service_ready",
                "aggregate/tenant_count": len(specs),
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
                for spec in specs
            ]
        )
        rows = [
            [
                result.name,
                result.session_id,
                result.final_reward,
                result.final_loss,
                result.elapsed,
                result.train_tokens,
                result.sample_tokens,
                result.prefill_tokens,
                result.sampler_path,
            ]
            for result in results
        ]
        table = orchestrator.table(
            [
                "tenant",
                "session_id",
                "final_reward",
                "final_loss",
                "elapsed_seconds",
                "train_tokens",
                "sample_tokens",
                "prefill_tokens",
                "sampler_path",
            ],
            rows,
        )
        payload: dict[str, Any] = {
            "progress/phase": "all_tenants_finished",
            "aggregate/tenant_count": len(results),
            "aggregate/elapsed_seconds": time.time() - started,
            "aggregate/final_reward_mean": sum(result.final_reward for result in results) / len(results),
            "aggregate/tokens/train_total": sum(result.train_tokens for result in results),
            "aggregate/tokens/sample_total": sum(result.sample_tokens for result in results),
            "aggregate/tokens/prefill_total": sum(result.prefill_tokens for result in results),
        }
        if table is not None:
            payload["aggregate/results"] = table
        orchestrator.log(payload, step=int(settings.get("steps", 8)))
        for result in results:
            print(
                f"{result.name}: session={result.session_id} "
                f"reward={result.final_reward:.3f} loss={result.final_loss:.3f} sampler={result.sampler_path}"
            )
    finally:
        service_client.close()
        orchestrator.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lfm2_5_1_2b_1x_l4_multitenant_rl.yaml")
    args = parser.parse_args()

    asyncio.run(train(args))


if __name__ == "__main__":
    main()
