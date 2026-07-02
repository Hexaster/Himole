import torch

from himole.eval.generate import generate_answers


class _Batch(dict):
    def to(self, device):
        return _Batch({k: v.to(device) for k, v in self.items()})


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, prompts, return_tensors, truncation, padding=False):
        if isinstance(prompts, str):
            prompts = [prompts]
        max_len = max(len(p) for p in prompts)
        rows = []
        masks = []
        for prompt in prompts:
            ids = list(range(1, len(prompt) + 1))
            pad = [self.pad_token_id] * (max_len - len(ids))
            rows.append(ids + pad)
            masks.append([1] * len(ids) + [0] * len(pad))
        return _Batch({
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.tensor(masks),
        })

    def decode(self, token_ids, skip_special_tokens=True):
        return "answer"


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.generate_calls = 0
        self.batch_sizes = []

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, pad_token_id):
        self.generate_calls += 1
        self.batch_sizes.append(input_ids.shape[0])
        return torch.cat([input_ids, torch.ones(input_ids.shape[0], 1, dtype=input_ids.dtype)], dim=1)


def test_generate_answers_batches_prompts():
    model = _Model()
    tokenizer = _Tokenizer()

    outputs = generate_answers(model, tokenizer, ["a", "bb", "ccc", "dddd", "eeeee"], batch_size=2)

    assert outputs == ["answer"] * 5
    assert model.generate_calls == 3
    assert model.batch_sizes == [2, 2, 1]
