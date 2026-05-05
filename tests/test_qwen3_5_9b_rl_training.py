import asyncio
import importlib.util
import py_compile
import sys
from pathlib import Path

from ray_unsloth import ModelInput
from ray_unsloth.clients.sampling import SamplingClient
from ray_unsloth.config import RuntimeConfig
from ray_unsloth.types import GeneratedSequence, SampleResponse, SamplingParams


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "qwen3_5_9b_rl_training.py"
CONFIG_PATH = Path(__file__).parents[1] / "configs" / "qwen3_5_9b_2x_l4_sharded.yaml"
SPEC = importlib.util.spec_from_file_location("qwen3_5_9b_rl_training", EXAMPLE_PATH)
assert SPEC is not None
qwen3_5_9b_rl_training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen3_5_9b_rl_training
assert SPEC.loader is not None
SPEC.loader.exec_module(qwen3_5_9b_rl_training)


def test_qwen3_5_9b_rl_example_is_valid_python():
    py_compile.compile(str(EXAMPLE_PATH), doraise=True)


def test_qwen3_5_9b_rl_example_uses_sharded_config_defaults():
    source = EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "BASE_MODEL = \"qwen3.5-9b-instruct\"" in source
    assert "MIN_TRAIN_DATUMS = 1" in source
    assert "configs/qwen3_5_9b_2x_l4_sharded.yaml" in source
    assert "ServiceClient(config=args.config)" in source
    assert 'loss_fn="importance_sampling"' in source
    assert "save_weights_and_get_sampling_client" not in source
    assert "live_training_sampling_client" in source


def test_qwen3_5_9b_2x_l4_config_selects_modal_sharded_model():
    config = RuntimeConfig.from_file(CONFIG_PATH)
    settings = qwen3_5_9b_rl_training.load_local_settings(CONFIG_PATH)

    assert config.modal.enabled is True
    assert config.modal.gpu == "L4:2"
    assert config.distributed.enabled is False
    assert config.distributed.mode is None
    assert config.model.base_model == "unsloth/Qwen3.5-9B"
    assert config.model.max_seq_length == 2048
    assert config.model.load_in_4bit is False
    assert config.model.fast_inference is False
    assert config.model.device_map == "auto"
    assert config.lora.rank == 16
    assert config.lora.alpha == 16
    assert settings["min_train_datums"] == 1
    assert settings["batch_size"] == 2
    assert settings["group_size"] == 4
    assert settings["max_tokens"] == 96
    assert settings["temperature"] == 0.8
    assert settings["top_p"] == 0.95
    assert settings["exploration_temperature"] == 1.0
    assert settings["exploration_top_p"] == 0.95
    assert settings["sft_anchor_weight"] == 0.2
    assert settings["wandb"]["enabled"] is True
    assert settings["wandb"]["project"] == "ray-unsloth-rl"
    assert settings["wandb"]["log_completions"] == 64


def test_qwen3_5_9b_reward_has_nonbinary_signal_for_rl():
    correct_short = qwen3_5_9b_rl_training.grade_answer("So \\boxed{732}", "732")
    correct_long = qwen3_5_9b_rl_training.grade_answer(
        "I will compute this carefully over many words and then provide \\boxed{732}",
        "732",
    )
    nearby_wrong = qwen3_5_9b_rl_training.grade_answer("So \\boxed{733}", "732")
    unboxed_correct = qwen3_5_9b_rl_training.grade_answer("The answer is 732", "732")
    unparseable = qwen3_5_9b_rl_training.grade_answer("I cannot determine it.", "732")

    assert correct_short > correct_long > unboxed_correct > nearby_wrong > unparseable
    assert nearby_wrong > 0.0


def test_qwen3_5_9b_problems_are_less_trivial_than_base_rl_example():
    questions = [problem.question for problem in qwen3_5_9b_rl_training.PROBLEMS]

    assert qwen3_5_9b_rl_training.FEWSHOT_PREFIX
    assert any("\\boxed{100}" in message["content"] for message in qwen3_5_9b_rl_training.FEWSHOT_PREFIX)
    assert any("remainder" in question for question in questions)
    assert any("ratio" in question for question in questions)
    assert "A box has 8 red marbles" not in "\n".join(questions)


class _FakeSamplingClient:
    def __init__(self):
        self.calls: list[SamplingParams] = []

    async def sample_async(self, prompt, num_samples, sampling_params):
        del prompt
        self.calls.append(sampling_params)
        if len(self.calls) <= 1:
            sequences = [
                GeneratedSequence(tokens=[1], text="\\boxed{732}", logprobs=[-0.1]),
                GeneratedSequence(tokens=[2], text="\\boxed{732}", logprobs=[-0.2]),
            ]
        else:
            sequences = [
                GeneratedSequence(tokens=[1], text="\\boxed{732}", logprobs=[-0.1]),
                GeneratedSequence(tokens=[2], text="\\boxed{733}", logprobs=[-0.2]),
            ]
        return SampleResponse(sequences=sequences)


def test_qwen3_5_9b_rollout_uses_absolute_advantages_for_degenerate_bad_groups(monkeypatch):
    monkeypatch.setattr(
        qwen3_5_9b_rl_training,
        "build_generation_prompt",
        lambda tokenizer, problem: ModelInput.from_ints([101, 102]),
    )
    class BadSamplingClient:
        async def sample_async(self, prompt, num_samples, sampling_params):
            del prompt, num_samples, sampling_params
            return SampleResponse(
                sequences=[
                    GeneratedSequence(tokens=[1], text="I cannot determine it.", logprobs=[-0.1]),
                    GeneratedSequence(tokens=[2], text="I cannot determine it.", logprobs=[-0.2]),
                ]
            )

    client = BadSamplingClient()
    problem = qwen3_5_9b_rl_training.MathProblem("Compute 37 * 24 - 156.", "732")

    async def run():
        return await qwen3_5_9b_rl_training.collect_rollouts(
            tokenizer=None,
            sampling_client=client,
            problems=[problem],
            group_size=2,
            sampling_params=SamplingParams(max_tokens=8, temperature=0.8, top_p=0.95),
        )

    rollouts = asyncio.run(run())
    datums = [datum for rollout in rollouts for datum in qwen3_5_9b_rl_training.rollout_to_datums(rollout)]

    assert rollouts[0].degenerate is True
    assert len(datums) == 2
    assert datums[0].loss_fn_inputs["advantages"].tolist()[-1] < 0.0


def test_qwen3_5_9b_rollout_group_relative_advantages_when_rewards_vary(monkeypatch):
    monkeypatch.setattr(
        qwen3_5_9b_rl_training,
        "build_generation_prompt",
        lambda tokenizer, problem: ModelInput.from_ints([101, 102]),
    )
    client = _FakeSamplingClient()
    problem = qwen3_5_9b_rl_training.MathProblem("Compute 37 * 24 - 156.", "732")

    async def run():
        return await qwen3_5_9b_rl_training.collect_rollouts(
            tokenizer=None,
            sampling_client=client,
            problems=[problem],
            group_size=2,
            sampling_params=SamplingParams(max_tokens=8, temperature=0.8, top_p=0.95),
        )

    asyncio.run(run())
    rollouts = asyncio.run(run())
    datums = [datum for rollout in rollouts for datum in qwen3_5_9b_rl_training.rollout_to_datums(rollout)]

    assert rollouts[0].degenerate is False
    assert len(datums) == 2
    assert client.calls[0].temperature == 0.8


def test_live_training_sampling_client_reuses_training_actor_without_saving():
    actor = object()

    class FakeTrainingClient:
        session_id = "train-123"
        _actor = actor

    sampler = qwen3_5_9b_rl_training.live_training_sampling_client(FakeTrainingClient(), name="rl")

    assert isinstance(sampler, SamplingClient)
    assert sampler.session_id == "train-123-rl"
    assert sampler._actors == [actor]


class _TinyTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=False):
        del messages, tokenize, add_generation_prompt, enable_thinking
        return [10, 11, 12]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) % 50 + 20 for char in text]


def test_supervised_anchor_datum_teaches_boxed_gold_answer():
    problem = qwen3_5_9b_rl_training.MathProblem("Compute 37 * 24 - 156.", "732")

    datum = qwen3_5_9b_rl_training.build_supervised_datum(
        tokenizer=_TinyTokenizer(),
        problem=problem,
        weight=0.2,
    )

    target_tokens = datum.loss_fn_inputs["target_tokens"].tolist()
    weights = datum.loss_fn_inputs["weights"].tolist()

    assert target_tokens[-1] == 99
    assert weights[:2] == [0.0, 0.0]
    assert all(weight == 0.2 for weight in weights[2:])
    assert datum.model_input.length == len(target_tokens)


def test_wandb_progress_logs_multiple_events_for_one_training_step():
    class FakeRun:
        def __init__(self):
            self.calls = []

        def log(self, payload, step):
            self.calls.append((payload, step))

    run = FakeRun()
    logger = qwen3_5_9b_rl_training.WandbLogger(enabled=True, run=run)

    logger.log_progress("sampling_started", step=0, extra={"data/problem_count": 2})
    logger.log_progress("rollouts_collected", step=0, extra={"data/datums": 1})

    assert [step for _, step in run.calls] == [1, 2]
    assert [payload["train/step"] for payload, _ in run.calls] == [0, 0]
    assert run.calls[0][0]["progress/sampling_started"] == 1.0
    assert run.calls[1][0]["progress/rollouts_collected"] == 1.0
    assert run.calls[1][0]["data/datums"] == 1


class _FakeWandbLogger:
    max_completion_rows = 64

    def table(self, columns, rows):
        return {"columns": columns, "rows": rows}

    def histogram(self, values):
        return {"histogram": list(values)}


def test_wandb_step_payload_logs_prescriptive_rl_metrics():
    problem = qwen3_5_9b_rl_training.MathProblem("Compute 37 * 24 - 156.", "732")
    rollout = qwen3_5_9b_rl_training.ProblemRollout(
        problem=problem,
        prompt=ModelInput.from_ints([101, 102]),
        completions=[
            qwen3_5_9b_rl_training.GradedCompletion(
                tokens=[201, 202],
                logprobs=[-0.1, -0.2],
                text="\\boxed{732}",
                reward=1.2,
                advantage=0.4,
            ),
            qwen3_5_9b_rl_training.GradedCompletion(
                tokens=[203],
                logprobs=[-0.3],
                text="\\boxed{733}",
                reward=0.4,
                advantage=-0.4,
            ),
        ],
        mean_reward=0.8,
        degenerate=False,
    )
    datum = qwen3_5_9b_rl_training.build_policy_datum(
        prompt=ModelInput.from_ints([101, 102]),
        completion_tokens=[201],
        completion_logprobs=[-0.1],
        advantage=0.4,
    )

    payload = qwen3_5_9b_rl_training.wandb_step_payload(
        step=3,
        rollouts=[rollout],
        datums=[datum],
        expert_datums=[datum],
        used_exploration_retry=True,
        mean_logprob=-0.25,
        mean_ratio=1.1,
        training_loss=0.7,
        rl_loss=0.2,
        sft_loss=0.5,
        optimizer_step=4,
        elapsed=12.5,
        learning_rate=4e-5,
        sampling_params=SamplingParams(max_tokens=8, temperature=0.8, top_p=0.95),
        logger=_FakeWandbLogger(),
    )

    assert payload["reward/mean"] == 0.8
    assert payload["reward/min"] == 0.4
    assert payload["reward/max"] == 1.2
    assert payload["rollout/completion_count"] == 2
    assert payload["rollout/used_exploration_retry"] == 1.0
    assert payload["policy/mean_logprob"] == -0.25
    assert payload["policy/mean_ratio"] == 1.1
    assert payload["data/datums"] == 1
    assert payload["data/rl_datums"] == 1
    assert payload["data/expert_datums"] == 1
    assert payload["sampling/temperature"] == 0.8
    assert payload["train/loss"] == 0.7
    assert payload["train/rl_loss"] == 0.2
    assert payload["train/sft_anchor_loss"] == 0.5
    assert payload["train/updated"] == 1.0
    assert payload["train/optimizer_step"] == 4
    assert payload["train/learning_rate"] == 4e-5
    assert payload["rollout/completions"]["rows"][0][0] == 3
    assert "\\boxed{732}" in payload["rollout/completions"]["rows"][0][-1]
