"""Entry point for the real A100 run: build config -> data -> model -> train -> eval OOD.

Usage:
    python scripts/train_baseline.py                 # full run on Qwen2.5-7B
    python scripts/train_baseline.py --tiny --steps 5 --samples 8   # smoke test
"""
import argparse


def require_full_run_cuda(cfg, device: str) -> None:
    """Prevent accidental 7B runs on CPU; tiny smoke tests may use CPU."""
    if not cfg.use_tiny and device != "cuda":
        raise RuntimeError("CUDA is required for the full Qwen2.5-7B run; use --tiny for CPU smoke tests.")


def parse_args(argv=None):
    """Parse baseline training options."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="use tiny-gpt2 for a fast smoke test")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import torch
    from datasets import load_dataset

    from himole.config import BaselineConfig
    from himole.utils import set_seed
    from himole.model.baseline_lora import load_tokenizer, load_base_model, attach_lora
    from himole.data.squad import load_squad, build_prompt
    from himole.data.newsqa import load_newsqa
    from himole.train.loop import train
    from himole.eval.generate import evaluate

    cfg = BaselineConfig()
    if args.tiny:
        cfg.use_tiny = True
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.samples is not None:
        cfg.max_train_samples = args.samples
        cfg.max_eval_samples = args.samples
    set_seed(cfg.seed)

    tokenizer = load_tokenizer(cfg)
    base_model = load_base_model(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    require_full_run_cuda(cfg, device)
    base_model.to(device)

    train_ds, _ = load_squad(tokenizer, cfg)

    # Build (prompt, gold) eval pairs for ID validation from raw SQuAD val split.
    raw_val = load_dataset(cfg.id_dataset)["validation"]
    if cfg.max_eval_samples:
        raw_val = raw_val.select(range(min(len(raw_val), cfg.max_eval_samples)))
    id_pairs = [{"prompt": build_prompt(e["context"], e["question"]),
                 "gold": e["answers"]["text"][0]} for e in raw_val]
    ood_pairs = load_newsqa(tokenizer, cfg)

    print("BASE MODEL ID (SQuAD):", evaluate(base_model, tokenizer, id_pairs, cutoff_len=cfg.cutoff_len, batch_size=cfg.eval_batch_size))
    print("BASE MODEL OOD (NewsQA):", evaluate(base_model, tokenizer, ood_pairs, cutoff_len=cfg.cutoff_len, batch_size=cfg.eval_batch_size))

    model = attach_lora(base_model, cfg)
    result = train(model, tokenizer, train_ds, id_pairs, cfg)
    print("LORA TRAIN RESULT:", result)

    print("LORA OOD (NewsQA):", evaluate(model, tokenizer, ood_pairs, cutoff_len=cfg.cutoff_len, batch_size=cfg.eval_batch_size))


if __name__ == "__main__":
    main()
