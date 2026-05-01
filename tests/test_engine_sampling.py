import sys
from types import SimpleNamespace

import torch

from ray_unsloth.config import LoRAConfig, ModelConfig
from ray_unsloth.runtime.unsloth.engine import UnslothEngine
from ray_unsloth.types import Datum, ModelInput, SamplingParams, TensorData


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

    UnslothEngine(
        session_id="train-1",
        model_config=ModelConfig(
            base_model="LiquidAI/LFM2.5-1.2B-Instruct",
            max_seq_length=4096,
            load_in_4bit=False,
            fast_inference=False,
            gpu_memory_utilization=0.75,
            trust_remote_code=False,
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
