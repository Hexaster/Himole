"""Tests for batched generation plumbing."""
import torch

from himole.eval.generate import generate_answers  # noqa: E402


class _FakeBatch(dict):
    def to(self, device):
        self["input_ids"] = self["input_ids"].to(device)
        self["attention_mask"] = self["attention_mask"].to(device)
        return self


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.padding_side = "right"
        self.truncation_side = "right"
        self.calls = []

    def __call__(self, batch, **kwargs):
        self.calls.append(
            {
                "batch": batch,
                "kwargs": kwargs,
                "padding_side": self.padding_side,
                "truncation_side": self.truncation_side,
            }
        )
        return _FakeBatch(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )

    def decode(self, tokens, skip_special_tokens=True):
        return "answer"


class _FakeModel:
    def __init__(self):
        self.param = torch.nn.Parameter(torch.zeros(1))

    def eval(self):
        pass

    def parameters(self):
        yield self.param

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4]])


def test_generate_answers_caps_prompt_to_cutoff_and_restores_tokenizer_state():
    tokenizer = _FakeTokenizer()

    out = generate_answers(
        _FakeModel(),
        tokenizer,
        ["long prompt"],
        cutoff_len=1024,
        batch_size=1,
    )

    assert out == ["answer"]
    assert tokenizer.calls[0]["kwargs"]["truncation"] is True
    assert tokenizer.calls[0]["kwargs"]["max_length"] == 1024
    assert tokenizer.calls[0]["padding_side"] == "left"
    assert tokenizer.calls[0]["truncation_side"] == "left"
    assert tokenizer.padding_side == "right"
    assert tokenizer.truncation_side == "right"
