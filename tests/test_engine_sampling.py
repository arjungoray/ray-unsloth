import sys
from types import SimpleNamespace

from ray_unsloth.runtime.unsloth.engine import UnslothEngine
from ray_unsloth.types import ModelInput, SamplingParams


class FakeRow:
    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeTensor:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def __getitem__(self, index):
        if isinstance(index, int):
            return FakeRow(self.rows[index])
        row, column = index
        return self.rows[row][column]

    def __iter__(self):
        for row in self.rows:
            yield FakeRow(row)


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    long = "long"

    @staticmethod
    def tensor(rows, dtype=None, device=None):
        del dtype, device
        return FakeTensor(rows)

    @staticmethod
    def manual_seed(seed):
        del seed

    @staticmethod
    def no_grad():
        return FakeNoGrad()


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

    def generate(self, input_ids, **kwargs):
        del kwargs
        return SimpleNamespace(sequences=FakeTensor([[*input_ids[0].tolist(), 11, 12]]), scores=[])


class FakeFastLanguageModel:
    @staticmethod
    def for_inference(model):
        del model


class FakeStoppingCriteria:
    pass


class FakeStoppingCriteriaList(list):
    pass


def test_sample_returns_generated_tokens_and_prompt_logprobs(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=FakeFastLanguageModel))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            StoppingCriteria=FakeStoppingCriteria,
            StoppingCriteriaList=FakeStoppingCriteriaList,
        ),
    )
    engine = UnslothEngine.__new__(UnslothEngine)
    engine.model = FakeModel()
    engine.tokenizer = FakeTokenizer()
    engine._generated_logprobs = lambda outputs: [[-0.1, -0.2]]
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
