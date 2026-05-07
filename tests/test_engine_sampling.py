import sys
from types import SimpleNamespace

import torch

from ray_unsloth.config import LoRAConfig, ModelConfig
from ray_unsloth.runtime.unsloth.engine import UnslothEngine
from ray_unsloth.types import AdamParams, Datum, ModelInput, SamplingParams, TensorData


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == ".":
            return [12]
        return [ord(char) for char in text]

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return "".join({11: "A", 12: "."}.get(token, "?") for token in tokens)


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.forward_input_ids = []
        self.forward_attention_masks = []
        self.next_tokens = [11, 12]

    def __call__(self, input_ids, attention_mask=None):
        self.forward_input_ids.append(input_ids.detach().cpu().tolist())
        self.forward_attention_masks.append(attention_mask.detach().cpu().tolist())
        token = self.next_tokens.pop(0)
        logits = torch.full((1, input_ids.shape[1], 100), -100.0)
        logits[0, -1, token] = 0.0
        return SimpleNamespace(logits=logits)


class FakeGenerateModel:
    device = "cpu"

    def __init__(self):
        self.generate_kwargs = None
        self.forward_input_ids = []
        self.forward_attention_masks = []

    def __call__(self, input_ids, attention_mask=None):
        self.forward_input_ids.append(input_ids.detach().cpu().tolist())
        self.forward_attention_masks.append(attention_mask.detach().cpu().tolist())
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], 100), -20.0)
        logits[:, :, 11] = 0.0
        logits[:, :, 12] = -5.0
        return SimpleNamespace(logits=logits)

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        scores = []
        for token in (11, 12):
            score = torch.full((2, 100), -100.0)
            score[:, 77] = 0.0
            scores.append(score)
        return SimpleNamespace(
            sequences=torch.tensor(
                [
                    [10, 11, 11, 12],
                    [10, 11, 12, 99],
                ],
                dtype=torch.long,
            ),
            scores=scores,
        )


class FakeFastLanguageModel:
    @staticmethod
    def for_inference(model):
        del model

    @staticmethod
    def for_training(model):
        del model


class FakeLossModel:
    device = "cpu"

    def __call__(self, input_ids, attention_mask=None):
        del attention_mask
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], 8), -10.0)
        logits[:, :, 3] = 0.0
        logits[:, :, 4] = -0.5
        return SimpleNamespace(logits=logits)


class LogitsToKeepLossModel:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, attention_mask=None, logits_to_keep=None):
        del attention_mask
        self.calls.append({"input_shape": tuple(input_ids.shape), "logits_to_keep": logits_to_keep})
        keep = int(logits_to_keep or input_ids.shape[1])
        logits = torch.full((input_ids.shape[0], keep, 8), -10.0)
        logits[:, :, 3] = 0.0
        logits[:, :, 4] = -0.5
        return SimpleNamespace(logits=logits)


class TinyTrainablePolicyModel(torch.nn.Module):
    @property
    def device(self):
        return self.token_logits.device

    def __init__(self):
        super().__init__()
        self.token_logits = torch.nn.Parameter(torch.zeros(8))

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        logits = self.token_logits.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


def test_load_model_uses_model_specific_unsloth_and_lora_config(monkeypatch, tmp_path):
    calls = {}

    class LoadingFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["from_pretrained"] = kwargs
            return FakeModel(), FakeTokenizer()

        @staticmethod
        def get_peft_model(model, **kwargs):
            calls["get_peft_model"] = kwargs
            return model

    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=LoadingFastLanguageModel))

    engine = UnslothEngine(
        session_id="train-1",
        model_config=ModelConfig(
            base_model="LiquidAI/LFM2.5-1.2B-Instruct",
            max_seq_length=4096,
            load_in_4bit=False,
            fast_inference=False,
            gpu_memory_utilization=0.75,
            trust_remote_code=False,
            attn_implementation="sdpa",
        ),
        lora_config=LoRAConfig(
            rank=16,
            alpha=16,
            dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
            loftq_config={"bits": 4},
        ),
        checkpoint_root=str(tmp_path),
    )

    assert calls["from_pretrained"]["model_name"] == "LiquidAI/LFM2.5-1.2B-Instruct"
    assert calls["from_pretrained"]["max_seq_length"] == 4096
    assert calls["from_pretrained"]["load_in_4bit"] is False
    assert calls["from_pretrained"]["fast_inference"] is False
    assert calls["from_pretrained"]["gpu_memory_utilization"] == 0.75
    assert calls["from_pretrained"]["trust_remote_code"] is False
    assert calls["from_pretrained"]["attn_implementation"] == "sdpa"
    assert "device_map" not in calls["from_pretrained"]
    assert calls["get_peft_model"] == {
        "r": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        "use_rslora": True,
        "loftq_config": {"bits": 4},
    }


def test_load_model_passes_configured_device_map(monkeypatch, tmp_path):
    calls = {}

    class LoadingFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["from_pretrained"] = kwargs
            return FakeModel(), FakeTokenizer()

        @staticmethod
        def get_peft_model(model, **kwargs):
            calls["get_peft_model"] = kwargs
            return model

    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=LoadingFastLanguageModel))

    UnslothEngine(
        session_id="train-sharded",
        model_config=ModelConfig(device_map="auto"),
        lora_config=LoRAConfig(),
        checkpoint_root=str(tmp_path),
    )

    assert calls["from_pretrained"]["device_map"] == "auto"


def test_distributed_load_model_pins_device_map_to_local_rank(monkeypatch, tmp_path):
    calls = {}

    class LoadingFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["from_pretrained"] = kwargs
            return FakeModel(), FakeTokenizer()

        @staticmethod
        def get_peft_model(model, **kwargs):
            calls["get_peft_model"] = kwargs
            return model

    class FakeDDP:
        def __init__(self, module, **kwargs):
            self.module = module
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=LoadingFastLanguageModel))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)

    engine = UnslothEngine(
        session_id="train-ddp",
        model_config=ModelConfig(load_in_4bit=True),
        lora_config=LoRAConfig(),
        checkpoint_root=str(tmp_path),
        rank=1,
        local_rank=0,
        world_size=2,
        init_method="tcp://127.0.0.1:12345",
    )

    assert calls["from_pretrained"]["device_map"] == {"": 0}
    assert isinstance(engine.model, FakeDDP)
    assert engine.model.kwargs["broadcast_bucket_size"] == 25 * 1024 * 1024


def test_sample_returns_generated_tokens_and_prompt_logprobs(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=FakeFastLanguageModel))
    engine = UnslothEngine.__new__(UnslothEngine)
    engine.model = FakeModel()
    engine.tokenizer = FakeTokenizer()
    engine._prompt_logprobs = lambda prompt_tokens, **kwargs: ([None, -0.3], [[], [(11, -0.3)]])

    response = engine.sample(
        ModelInput.from_ints([10, 11]),
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=2, temperature=0.0, stop=["."]),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=2,
    )

    assert response.sequences[0].tokens == [11]
    assert response.sequences[0].text == "A"
    assert response.sequences[0].stop_reason == "stop"
    assert response.sequences[0].logprobs is not None
    assert response.prompt_logprobs is not None
    assert len(response.prompt_logprobs) == 2
    assert response.topk_prompt_logprobs is not None
    assert len(response.topk_prompt_logprobs) == 2
    assert engine.model.forward_input_ids[0] == [[10, 11]]
    assert engine.model.forward_input_ids[1] == [[10, 11, 11]]
    assert engine.model.forward_attention_masks[0] == [[1, 1]]
    assert engine.model.forward_attention_masks[1] == [[1, 1, 1]]


def test_sample_uses_batched_generate_when_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=FakeFastLanguageModel))
    engine = UnslothEngine.__new__(UnslothEngine)
    engine.model = FakeGenerateModel()
    engine.tokenizer = FakeTokenizer()
    engine._prompt_logprobs = lambda prompt_tokens, **kwargs: (None, None)

    response = engine.sample(
        ModelInput.from_ints([10, 11]),
        num_samples=2,
        sampling_params=SamplingParams(max_tokens=2, temperature=0.8, top_p=0.9, stop=["."], max_time=12.5),
    )

    assert engine.model.generate_kwargs["num_return_sequences"] == 2
    assert engine.model.generate_kwargs["max_new_tokens"] == 2
    assert engine.model.generate_kwargs["temperature"] == 0.8
    assert engine.model.generate_kwargs["top_p"] == 0.9
    assert engine.model.generate_kwargs["max_time"] == 12.5
    assert engine.model.generate_kwargs["output_scores"] is False
    assert [sequence.tokens for sequence in response.sequences] == [[11], []]
    assert response.sequences[0].stop_reason == "stop"
    assert response.sequences[0].logprobs is not None
    assert len(response.sequences[0].logprobs) == 1
    assert response.sequences[0].logprobs[0] > -1.0
    assert engine.model.forward_input_ids == [[[10, 11, 11]]]
    assert engine.model.forward_attention_masks == [[[1, 1, 1]]]


def test_sample_caps_logprob_recompute_tokens_when_requested(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=FakeFastLanguageModel))
    engine = UnslothEngine.__new__(UnslothEngine)
    engine.model = FakeGenerateModel()
    engine.tokenizer = FakeTokenizer()
    engine._prompt_logprobs = lambda prompt_tokens, **kwargs: (None, None)

    response = engine.sample(
        ModelInput.from_ints([10, 11]),
        num_samples=2,
        sampling_params=SamplingParams(
            max_tokens=2,
            temperature=0.8,
            top_p=0.9,
            stop=[],
            logprobs_max_tokens=1,
        ),
    )

    assert [sequence.tokens for sequence in response.sequences] == [[11, 12], [12, 99]]
    assert [len(sequence.logprobs or []) for sequence in response.sequences] == [1, 1]
    assert engine.model.generate_kwargs["output_scores"] is True
    assert engine.model.forward_input_ids == []
    assert engine.model.forward_attention_masks == []


def test_optimizer_reuses_state_when_params_change():
    engine = UnslothEngine.__new__(UnslothEngine)
    engine.model = torch.nn.Linear(2, 1)
    engine.optimizer = None

    optimizer = engine._ensure_optimizer(AdamParams(learning_rate=1e-4))
    same_optimizer = engine._ensure_optimizer(AdamParams(learning_rate=2e-4, beta1=0.8, beta2=0.9))

    assert same_optimizer is optimizer
    assert same_optimizer.param_groups[0]["lr"] == 2e-4
    assert same_optimizer.param_groups[0]["betas"] == (0.8, 0.9)


def test_importance_sampling_backward_and_optim_step_increase_positive_advantage_logit(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=FakeFastLanguageModel))
    engine = UnslothEngine.__new__(UnslothEngine)
    engine.model = TinyTrainablePolicyModel()
    engine.tokenizer = FakeTokenizer()
    engine.optimizer = None
    engine.world_size = 1
    engine.step = 0
    datum = Datum(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[0, 3], dtype="int64", shape=[2]),
            "logprobs": TensorData(data=[0.0, -2.0], dtype="float32", shape=[2]),
            "advantages": TensorData(data=[0.0, 1.0], dtype="float32", shape=[2]),
            "weights": TensorData(data=[0.0, 1.0], dtype="float32", shape=[2]),
        },
    )

    before = float(engine.model.token_logits[3].detach())
    engine.forward_backward([datum], loss_fn="importance_sampling")
    result = engine.optim_step(AdamParams(learning_rate=0.1))
    after = float(engine.model.token_logits[3].detach())

    assert result.step == 1
    assert after > before


def test_cross_entropy_accepts_tinker_target_tokens_and_weights():
    engine = UnslothEngine.__new__(UnslothEngine)
    engine.model = FakeLossModel()
    engine.tokenizer = FakeTokenizer()
    datum = Datum(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[3, 4], dtype="int64", shape=[2]),
            "weights": TensorData(data=[0.0, 1.0], dtype="float32", shape=[2]),
        },
    )

    loss, _outputs, logprobs = engine._cross_entropy_loss([datum])
    loss_outputs = engine._loss_fn_outputs(logprobs)

    assert loss.item() > 0
    assert loss_outputs[0]["logprobs"].tolist()[0] == 0.0
    assert loss_outputs[0]["logprobs"].tolist()[1] < 0.0


def test_policy_loss_returns_logprobs_and_ratios():
    engine = UnslothEngine.__new__(UnslothEngine)
    engine.model = FakeLossModel()
    engine.tokenizer = FakeTokenizer()
    datum = Datum(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[3, 4], dtype="int64", shape=[2]),
            "logprobs": TensorData(data=[0.0, -1.0], dtype="float32", shape=[2]),
            "advantages": TensorData(data=[0.0, 1.0], dtype="float32", shape=[2]),
        },
    )

    loss, _outputs, logprobs, ratios = engine._policy_loss([datum], loss_fn="importance_sampling")
    loss_outputs = engine._loss_fn_outputs(logprobs, ratios=ratios)

    assert torch.isfinite(loss)
    assert loss_outputs[0]["logprobs"].tolist()[1] < 0.0
    assert loss_outputs[0]["ratios"].tolist()[1] > 0.0


def test_policy_loss_uses_logits_to_keep_for_completion_suffix():
    engine = UnslothEngine.__new__(UnslothEngine)
    engine.model = LogitsToKeepLossModel()
    engine.tokenizer = FakeTokenizer()
    input_tokens = [1] * 1000
    datum = Datum(
        model_input=ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[0] * 998 + [3, 4], dtype="int64", shape=[1000]),
            "logprobs": TensorData(data=[0.0] * 998 + [-1.0, -1.0], dtype="float32", shape=[1000]),
            "advantages": TensorData(data=[0.0] * 998 + [1.0, 1.0], dtype="float32", shape=[1000]),
            "weights": TensorData(data=[0.0] * 998 + [1.0, 1.0], dtype="float32", shape=[1000]),
        },
    )

    loss, _outputs, logprobs, ratios = engine._policy_loss([datum], loss_fn="importance_sampling")

    assert torch.isfinite(loss)
    assert engine.model.calls == [{"input_shape": (1, 1000), "logits_to_keep": 2}]
    assert logprobs.shape == (1, 1000)
    assert ratios.shape == (1, 1000)
