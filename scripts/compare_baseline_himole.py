"""Train and compare Base, LoRA, and HiMoLE on shared ID/OOD examples."""
import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    """Parse comparison report options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200, help="training and evaluation sample cap")
    parser.add_argument("--out", default="outputs/himole_compare.json", help="JSON report path")
    parser.add_argument("--tiny", action="store_true", help="use tiny-gpt2 for a local smoke comparison")
    parser.add_argument("--steps", type=int, default=None, help="override both Stage-2 and LoRA steps")
    parser.add_argument("--stage1-steps", type=int, default=None, help="override HiMoLE Stage-1 steps")
    return parser.parse_args(argv)


def _parameter_counts(model):
    return {
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def main(argv=None):
    """Run all three model states against one set of prompts and metrics."""
    args = parse_args(argv)

    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    import torch
    from datasets import load_dataset

    from himole.config import BaselineConfig, HimoleConfig
    from himole.data.newsqa import load_newsqa
    from himole.data.squad import build_prompt, load_squad
    from himole.eval.generate import evaluate
    from himole.model.baseline_lora import attach_lora, load_base_model, load_tokenizer
    from himole.model.patching import attach_himole_to_model
    from himole.train.loop import train
    from himole.train.stage1 import initialize_kcgs
    from himole.train.stage2 import train_himole_stage2
    from himole.utils import set_seed

    baseline_cfg = BaselineConfig(
        use_tiny=args.tiny,
        max_train_samples=args.samples,
        max_eval_samples=args.samples,
    )
    himole_cfg = HimoleConfig(
        use_tiny=args.tiny,
        max_train_samples=args.samples,
        max_eval_samples=args.samples,
    )
    if args.steps is not None:
        baseline_cfg.max_steps = args.steps
        himole_cfg.max_steps = args.steps
    if args.stage1_steps is not None:
        himole_cfg.stage1_max_steps = args.stage1_steps
    if himole_cfg.use_tiny:
        himole_cfg.clustering_encoder = himole_cfg.tiny_model
    himole_cfg.validate()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.tiny and device != "cuda":
        raise RuntimeError("CUDA is required for the full comparison; use --tiny for a local smoke test.")

    set_seed(baseline_cfg.seed)
    tokenizer = load_tokenizer(baseline_cfg)
    train_ds, _ = load_squad(tokenizer, baseline_cfg)
    raw_val = load_dataset(baseline_cfg.id_dataset)["validation"]
    raw_val = raw_val.select(range(min(len(raw_val), args.samples)))
    id_pairs = [
        {"prompt": build_prompt(example["context"], example["question"]), "gold": example["answers"]["text"][0]}
        for example in raw_val
    ]
    ood_pairs = load_newsqa(tokenizer, baseline_cfg)

    report = {
        "eval_samples": {"id": len(id_pairs), "ood": len(ood_pairs)},
        "models": {},
    }

    base_model = load_base_model(baseline_cfg).to(device)
    base_counts = _parameter_counts(base_model)
    report["models"]["base"] = {
        "id": evaluate(base_model, tokenizer, id_pairs, baseline_cfg.cutoff_len, baseline_cfg.eval_batch_size),
        "ood": evaluate(base_model, tokenizer, ood_pairs, baseline_cfg.cutoff_len, baseline_cfg.eval_batch_size),
        "trainable_parameters": 0,
        "total_parameters": base_counts["total_parameters"],
    }

    lora_model = attach_lora(base_model, baseline_cfg)
    lora_result = train(lora_model, tokenizer, train_ds, id_pairs, baseline_cfg)
    report["models"]["lora"] = {
        "id": evaluate(lora_model, tokenizer, id_pairs, baseline_cfg.cutoff_len, baseline_cfg.eval_batch_size),
        "ood": evaluate(lora_model, tokenizer, ood_pairs, baseline_cfg.cutoff_len, baseline_cfg.eval_batch_size),
        "training": lora_result,
        **_parameter_counts(lora_model),
    }
    del lora_model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    set_seed(himole_cfg.seed)
    himole_model = attach_himole_to_model(load_base_model(himole_cfg).to(device), himole_cfg)
    initialize_kcgs(himole_model, train_ds, himole_cfg, tokenizer=tokenizer)
    himole_result = train_himole_stage2(himole_model, tokenizer, train_ds, himole_cfg, val_eval_pairs=id_pairs)
    report["models"]["himole"] = {
        "id": evaluate(himole_model, tokenizer, id_pairs, himole_cfg.cutoff_len, himole_cfg.eval_batch_size),
        "ood": evaluate(himole_model, tokenizer, ood_pairs, himole_cfg.cutoff_len, himole_cfg.eval_batch_size),
        "training": himole_result,
        **_parameter_counts(himole_model),
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote comparison report to {output_path}")


if __name__ == "__main__":
    main()
