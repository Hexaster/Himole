"""Tests for prompt construction and loss masking."""
from transformers import AutoTokenizer

from himole.config import HimoleConfig
from himole.data.squad import build_prompt, format_example, load_squad

TOK = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
if TOK.pad_token is None:
    TOK.pad_token = TOK.eos_token


def test_prompt_excludes_answer():
    p = build_prompt("Paris is the capital of France.", "What is the capital of France?")
    assert "France" in p and "Paris" in p
    assert "Answer:" in p          # template marks where the answer goes
    assert p.rstrip().endswith("Answer:")  # prompt stops before the answer text


def test_labels_mask_prompt_tokens():
    ex = {"context": "Paris is the capital of France.",
          "question": "What is the capital of France?",
          "answers": {"text": ["Paris"]}}
    out = format_example(ex, TOK, cutoff_len=64)
    assert set(out.keys()) >= {"input_ids", "attention_mask", "labels"}
    assert len(out["input_ids"]) == len(out["labels"])
    # at least one masked (prompt) and one unmasked (answer) label
    assert any(l == -100 for l in out["labels"])
    assert any(l != -100 for l in out["labels"])
    # every unmasked label must equal the corresponding input id (next-token target)
    for i, l in enumerate(out["labels"]):
        if l != -100:
            assert l == out["input_ids"][i]


def test_squad_loader_uses_separate_training_and_validation_caps(monkeypatch):
    class Split(list):
        column_names = ["context", "question", "answers"]

        def select(self, indices):
            return Split(self[index] for index in indices)

        def map(self, function, remove_columns):
            return Split(function(example) for example in self)

    examples = Split(
        {"context": "context", "question": "question", "answers": {"text": ["answer"]}}
        for _ in range(10)
    )
    monkeypatch.setattr(
        "himole.data.squad.load_dataset",
        lambda name: {"train": examples, "validation": examples},
    )
    cfg = HimoleConfig(max_train_samples=3, max_eval_samples=2)

    train, validation = load_squad(TOK, cfg)

    assert len(train) == 3
    assert len(validation) == 2
