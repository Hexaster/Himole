"""Entry point for the real A100 run: build config -> data -> model -> train -> eval OOD.

Usage:
    python scripts/train_baseline.py                 # full run on Llama-2-7B
    python scripts/train_baseline.py --tiny --steps 5 --samples 8   # smoke test
"""
import argparse

from datasets import load_dataset

from himole.config import BaselineConfig
from himole.utils import set_seed
from himole.model.baseline_lora import load_tokenizer, load_base_model, attach_lora
from himole.data.squad import load_squad, build_prompt
from himole.data.newsqa import load_newsqa
from himole.train.loop import train
from himole.eval.generate import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="use tiny-gpt2 for a fast smoke test")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    args = ap.parse_args()

    cfg = BaselineConfig()
    if args.tiny:
        cfg.use_tiny = True
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.samples is not None:
        cfg.max_train_samples = args.samples
    set_seed(cfg.seed)

    tokenizer = load_tokenizer(cfg)
    base_model = load_base_model(cfg)

    train_ds, _ = load_squad(tokenizer, cfg)

    # Build (prompt, gold) eval pairs for ID validation from raw SQuAD val split.
    raw_val = load_dataset(cfg.id_dataset)["validation"]
    if cfg.max_train_samples:
        raw_val = raw_val.select(range(cfg.max_train_samples))
    id_pairs = [{"prompt": build_prompt(e["context"], e["question"]),
                 "gold": e["answers"]["text"][0]} for e in raw_val]
    ood_pairs = load_newsqa(tokenizer, cfg)

    print("BASE MODEL ID (SQuAD):", evaluate(base_model, tokenizer, id_pairs))
    print("BASE MODEL OOD (NewsQA):", evaluate(base_model, tokenizer, ood_pairs))

    model = attach_lora(base_model, cfg)
    result = train(model, tokenizer, train_ds, id_pairs, cfg)
    print("LORA TRAIN RESULT:", result)

    print("LORA OOD (NewsQA):", evaluate(model, tokenizer, ood_pairs))


if __name__ == "__main__":
    main()
