import importlib.util
import py_compile
import sys
from pathlib import Path


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "qwen3_5_9b_math_dataset_rl_training.py"
CONFIG_PATH = Path(__file__).parents[1] / "configs" / "qwen3_5_9b_2x_l4_sharded.yaml"
SPEC = importlib.util.spec_from_file_location("qwen3_5_9b_math_dataset_rl_training", EXAMPLE_PATH)
assert SPEC is not None
qwen3_5_9b_math_dataset_rl_training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen3_5_9b_math_dataset_rl_training
assert SPEC.loader is not None
SPEC.loader.exec_module(qwen3_5_9b_math_dataset_rl_training)


def test_math_dataset_rl_example_is_valid_python():
    py_compile.compile(str(EXAMPLE_PATH), doraise=True)


def test_math_dataset_rl_example_uses_qwen3_5_model_and_cookbook_datasets():
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    config = CONFIG_PATH.read_text(encoding="utf-8")

    assert 'BASE_MODEL = "qwen3.5-9b-instruct"' in source
    assert "EleutherAI/hendrycks_math" in source
    assert "HuggingFaceH4/MATH-500" in source
    assert "openai/gsm8k" in source
    assert 'loss_fn="importance_sampling"' in source
    assert "create_live_sampling_client" in source
    assert "qwen3_5_9b_math_dataset_rl_training:" in config
    assert "dataset: math" in config
    assert "max_tokens: 8096" in config
    assert "max_train_tokens: 2048" in config
    assert "train_microbatch_size: 1" in config
    assert "top_k: 20" in config
    assert "max_time: 120.0" in config
    assert "enable_thinking: true" in config
    assert "batch_size: 1" in config
    assert "- <END>" in config


def test_extract_boxed_handles_nested_latex():
    assert qwen3_5_9b_math_dataset_rl_training.extract_boxed(r"The answer is \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"
    assert qwen3_5_9b_math_dataset_rl_training.extract_boxed("No boxed answer") is None


def test_extract_gsm8k_final_answer_strips_marker_and_commas():
    answer = "Natalia sold 48 clips in April.\n#### 1,234"

    assert qwen3_5_9b_math_dataset_rl_training.extract_gsm8k_final_answer(answer) == "1234"


def test_grade_math_answer_rewards_boxed_equivalent_fraction():
    reward = qwen3_5_9b_math_dataset_rl_training.grade_math_answer(
        r"We simplify to \boxed{0.5}.",
        r"\frac{1}{2}",
    )
    wrong = qwen3_5_9b_math_dataset_rl_training.grade_math_answer(
        r"We simplify to \boxed{0.25}.",
        r"\frac{1}{2}",
    )

    assert reward > wrong
    assert reward > 1.0


def test_grade_math_answer_accepts_symbolically_equivalent_polynomial():
    reward = qwen3_5_9b_math_dataset_rl_training.grade_math_answer(
        r"The remainder is \boxed{1 - x^5}.",
        r"-x^5 + 1",
    )

    assert reward > 1.0


def test_rows_to_problems_parses_math_and_gsm8k_rows():
    math_rows = [
        {"problem": "Find x.", "solution": r"Solving gives \boxed{7}."},
        {"problem": "Bad row.", "solution": "No final answer."},
    ]
    gsm8k_rows = [
        {"question": "How many?", "answer": "Work\n#### 42"},
    ]

    math_problems = qwen3_5_9b_math_dataset_rl_training._rows_to_problems(
        math_rows,
        dataset_name="math",
        limit=None,
    )
    gsm8k_problems = qwen3_5_9b_math_dataset_rl_training._rows_to_problems(
        gsm8k_rows,
        dataset_name="gsm8k",
        limit=None,
    )

    assert [(problem.question, problem.answer) for problem in math_problems] == [("Find x.", "7")]
    assert [(problem.question, problem.answer) for problem in gsm8k_problems] == [("How many?", "42")]


class _TinyTokenizer:
    eos_token_id = 99
    last_messages = None
    last_enable_thinking = None

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=False):
        del tokenize, add_generation_prompt
        self.last_messages = messages
        self.last_enable_thinking = enable_thinking
        return [10, 11, 12]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) % 50 + 20 for char in text]


def test_supervised_anchor_uses_cookbook_boxed_prompt():
    tokenizer = _TinyTokenizer()
    problem = qwen3_5_9b_math_dataset_rl_training.MathProblem("Simplify 1/4 + 1/4.", r"\frac{1}{2}")

    datum = qwen3_5_9b_math_dataset_rl_training.build_supervised_datum(
        tokenizer=tokenizer,
        problem=problem,
        weight=0.1,
    )

    assert tokenizer.last_messages[-1]["content"].endswith(r"Write your answer in \boxed{} format, then write <END>.")
    assert tokenizer.last_enable_thinking is True
    assert datum.loss_fn_inputs["target_tokens"].tolist()[-1] == 99
    assert datum.loss_fn_inputs["weights"].tolist()[-1] == 0.1


def test_rollout_to_datums_respects_training_token_cap():
    rollout = qwen3_5_9b_math_dataset_rl_training.ProblemRollout(
        problem=qwen3_5_9b_math_dataset_rl_training.MathProblem("Compute 1+1.", "2"),
        prompt=qwen3_5_9b_math_dataset_rl_training.ModelInput.from_ints([101, 102, 103]),
        completions=[
            qwen3_5_9b_math_dataset_rl_training.GradedCompletion(
                tokens=[201, 202, 203, 204],
                logprobs=[-0.1, -0.2, -0.3, -0.4],
                text=r"\boxed{2}",
                reward=1.2,
                advantage=0.5,
            )
        ],
        mean_reward=1.2,
        degenerate=False,
    )

    datums = qwen3_5_9b_math_dataset_rl_training.rollout_to_datums(rollout, max_train_tokens=5)

    assert len(datums) == 1
    assert datums[0].model_input.length == 5
    assert datums[0].loss_fn_inputs["target_tokens"].tolist() == [0, 0, 201, 202, 203]


def test_dataset_payload_logs_long_generation_controls():
    problem = qwen3_5_9b_math_dataset_rl_training.MathProblem("Compute 1+1.", "2")
    rollout = qwen3_5_9b_math_dataset_rl_training.ProblemRollout(
        problem=problem,
        prompt=qwen3_5_9b_math_dataset_rl_training.ModelInput.from_ints([1, 2]),
        completions=[
            qwen3_5_9b_math_dataset_rl_training.GradedCompletion(
                tokens=[3, 4],
                logprobs=[-0.1, -0.2],
                text="<think>ok</think> \\boxed{2}",
                reward=1.2,
                advantage=0.0,
            )
        ],
        mean_reward=1.2,
        degenerate=True,
    )

    class Logger:
        max_completion_rows = 64

        def table(self, columns, rows):
            return {"columns": columns, "rows": rows}

        def histogram(self, values):
            return {"histogram": values}

    payload = qwen3_5_9b_math_dataset_rl_training.dataset_wandb_step_payload(
        step=0,
        rollouts=[rollout],
        datums=[],
        expert_datums=[],
        used_exploration_retry=False,
        mean_logprob=0.0,
        mean_ratio=0.0,
        training_loss=0.0,
        rl_loss=0.0,
        sft_loss=0.0,
        optimizer_step=None,
        elapsed=1.0,
        learning_rate=4e-5,
        sampling_params=qwen3_5_9b_math_dataset_rl_training.SamplingParams(
            max_tokens=8096,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            max_time=120.0,
            logprobs_max_tokens=2048,
            stop=["<END>"],
        ),
        logger=Logger(),
        enable_thinking=True,
        token_totals={"prefill": 12, "sample": 34, "train": 56},
    )

    assert payload["rollout/boxed_fraction"] == 1.0
    assert payload["rollout/thinking_fraction"] == 1.0
    assert payload["rollout/max_token_fraction"] == 0.0
    assert payload["sampling/max_time"] == 120.0
    assert payload["sampling/logprobs_max_tokens"] == 2048
    assert payload["sampling/stop_sequence_count"] == 1
    assert payload["tokens/prefill_total"] == 12
    assert payload["tokens/sample_total"] == 34
    assert payload["tokens/train_total"] == 56
