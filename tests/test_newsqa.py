from himole.config import BaselineConfig
from himole.data.newsqa import load_newsqa


class _FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def select(self, indices):
        return _FakeDataset([self.rows[i] for i in indices])


def test_load_newsqa_uses_eval_sample_cap(monkeypatch):
    rows = [
        {"context": f"context {i}", "question": f"question {i}", "answers": [f"answer {i}"]}
        for i in range(4)
    ]

    def fake_load_dataset(name, split):
        assert name == "lucadiliello/newsqa"
        assert split == "validation"
        return _FakeDataset(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    cfg = BaselineConfig(max_train_samples=None, max_eval_samples=2)
    pairs = load_newsqa(None, cfg)

    assert len(pairs) == 2
    assert pairs[0]["gold"] == "answer 0"
    assert pairs[1]["gold"] == "answer 1"
