"""Run two-stage HiMoLE training on SQuAD, then report ID/OOD QA metrics."""
import argparse


def parse_args(argv=None):
    """Parse HiMoLE training options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true", help="use tiny-gpt2 for a short local smoke test")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--stage1-steps", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="examples per GPU pass; must divide the effective batch size of 16",
    )
    parser.add_argument(
        "--sequential-stage1",
        action="store_true",
        help="train KCGs on one GPU even when enough GPUs are visible",
    )
    parser.add_argument(
        "--resume-stage1",
        action="store_true",
        help="reuse compatible completed KCG checkpoints from stage1_output_dir",
    )
    stage_mode = parser.add_mutually_exclusive_group()
    stage_mode.add_argument(
        "--skip-stage1",
        action="store_true",
        help="skip KCG initialization and run an architecture smoke test with random experts",
    )
    stage_mode.add_argument(
        "--stage1-only",
        action="store_true",
        help="initialize KCG experts and stop before Stage 2",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Parse run flags, optionally initialize KCGs, and jointly train HiMoLE."""
    args = parse_args(argv)

    import torch
    from datasets import load_dataset

    from himole.config import HimoleConfig
    from himole.data.newsqa import load_newsqa
    from himole.data.squad import build_prompt, load_squad
    from himole.eval.generate import evaluate
    from himole.model.baseline_lora import load_base_model, load_tokenizer
    from himole.model.patching import attach_himole_to_model
    from himole.train.stage1 import (
        apply_stage1_initializations,
        initialize_kcgs,
        initialize_kcgs_parallel,
    )
    from himole.train.stage2 import train_himole_stage2
    from himole.utils import set_seed

    # This chunk builds the paper-shaped config and validates expert topology.
    cfg = HimoleConfig(use_tiny=args.tiny)
    if cfg.use_tiny:
        cfg.clustering_encoder = cfg.tiny_model
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.stage1_steps is not None:
        cfg.stage1_max_steps = args.stage1_steps
    if args.samples is not None:
        cfg.max_train_samples = args.samples
        cfg.max_eval_samples = args.samples
    if args.micro_batch_size is not None:
        cfg.micro_batch_size = args.micro_batch_size
    cfg.validate()
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not cfg.use_tiny and device != "cuda":
        raise RuntimeError("CUDA is required for the full Qwen2.5-7B run; use --tiny for a local smoke test.")

    tokenizer = load_tokenizer(cfg)
    train_ds, validation_ds = load_squad(tokenizer, cfg)

    parallel_stage1 = (
        not cfg.use_tiny
        and not args.skip_stage1
        and not args.sequential_stage1
        and torch.cuda.device_count() >= cfg.num_kcgs
    )
    if not cfg.use_tiny and not args.skip_stage1 and not args.sequential_stage1 and not parallel_stage1:
        print(
            f"Stage 1: {torch.cuda.device_count()} CUDA device(s) visible; "
            f"falling back to sequential training because {cfg.num_kcgs} are required"
        )

    parallel_state = None
    if parallel_stage1:
        print(f"Stage 1: training {cfg.num_kcgs} KCGs in parallel on {cfg.num_kcgs} GPUs")
        parallel_state = initialize_kcgs_parallel(
            train_ds,
            validation_ds,
            cfg,
            resume=args.resume_stage1,
        )
        if args.stage1_only:
            print("HIMOLE STAGE 1 RESULT:", {group_id: sorted(state) for group_id, state in parallel_state.items()})
            return

    model = attach_himole_to_model(load_base_model(cfg).to(device), cfg)

    if args.stage1_only:
        result = initialize_kcgs(
            model,
            train_ds,
            cfg,
            tokenizer=tokenizer,
            validation_dataset=validation_ds,
        )
        print("HIMOLE STAGE 1 RESULT:", {group_id: sorted(state) for group_id, state in result.items()})
        return

    raw_val = load_dataset(cfg.id_dataset)["validation"]
    if cfg.max_eval_samples is not None:
        raw_val = raw_val.select(range(min(len(raw_val), cfg.max_eval_samples)))
    id_pairs = [
        {"prompt": build_prompt(example["context"], example["question"]), "gold": example["answers"]["text"][0]}
        for example in raw_val
    ]
    ood_pairs = load_newsqa(tokenizer, cfg)

    if args.skip_stage1:
        print("Skipping Stage 1; using randomly initialized experts for this smoke test.")
    elif parallel_state is not None:
        apply_stage1_initializations(model, parallel_state)
    else:
        initialize_kcgs(
            model,
            train_ds,
            cfg,
            tokenizer=tokenizer,
            validation_dataset=validation_ds,
        )
    result = train_himole_stage2(model, tokenizer, train_ds, cfg, val_eval_pairs=id_pairs)
    print("HIMOLE TRAIN RESULT:", result)
    print("HIMOLE ID (SQuAD):", evaluate(model, tokenizer, id_pairs, cutoff_len=cfg.cutoff_len, batch_size=cfg.eval_batch_size))
    print("HIMOLE OOD (NewsQA):", evaluate(model, tokenizer, ood_pairs, cutoff_len=cfg.cutoff_len, batch_size=cfg.eval_batch_size))


if __name__ == "__main__":
    main()
