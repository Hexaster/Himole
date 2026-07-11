"""Load NewsQA as an OUT-OF-DISTRIBUTION eval set.

We only need raw (prompt, gold_answer) pairs for generation-time scoring here,
not loss-masked training tensors.
"""
from himole.data.squad import build_prompt


def load_newsqa(tokenizer, cfg):
    """Return a list of {"prompt", "gold"} dicts for OOD evaluation."""
    from datasets import load_dataset
    dataset = load_dataset(cfg.ood_dataset, split="validation")
    if cfg.max_eval_samples is not None:
        dataset = dataset.select(range(min(len(dataset), cfg.max_eval_samples)))
    return [{"prompt": build_prompt(ex["context"], ex["question"]),
             "gold": ex["answers"][0]} for ex in dataset]
