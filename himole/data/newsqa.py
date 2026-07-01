"""Load NewsQA as an OUT-OF-DISTRIBUTION eval set.

We only need raw (prompt, gold_answer) pairs for generation-time scoring here,
not loss-masked training tensors.
"""
from himole.data.squad import build_prompt


def load_newsqa(tokenizer, cfg):
    """Return a list of {"prompt", "gold"} dicts for OOD evaluation."""
    # TODO: pick a NewsQA source on the Hub (e.g. "lucadiliello/newsqa" or a mirror),
    #       map each example to {"prompt": build_prompt(context, question),
    #       "gold": first answer text}. Respect cfg.max_train_samples for smoke tests.
    #raise NotImplementedError("TODO: load NewsQA and return prompt/gold pairs")
    from datasets import load_dataset
    dataset = load_dataset("lucadiliello/newsqa", split="validation")
    if cfg.max_train_samples is not None:
        dataset = dataset.select(range(cfg.max_train_samples))
    id_pairs = [{'prompt': build_prompt(ex['context'], ex['question']),
                 'gold': ex['answers'][0]} for ex in dataset]
    return id_pairs
