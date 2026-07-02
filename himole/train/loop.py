"""A transparent, hand-written training loop so every step of LoRA fine-tuning
is visible: build batches -> forward -> loss -> backward -> optimizer step,
with periodic eval, early stopping, and best-checkpoint saving.

Multi-GPU note: pass an Accelerator instance from the launch script.
`accelerate launch --num_processes N scripts/train_baseline.py` handles process
spawning; this file just wraps the three objects (model/optimizer/loader) and
uses accelerator.backward() instead of loss.backward().
"""
import os

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from himole.eval.generate import evaluate
from himole.utils import get_logger

log = get_logger("train")


def collate(batch, tokenizer):
    """Pad a list of {input_ids,attention_mask,labels} to the batch's max length."""
    maxlen = max(len(b["input_ids"]) for b in batch)
    pad_id = tokenizer.pad_token_id

    def pad(seq, value):
        return seq + [value] * (maxlen - len(seq))

    input_ids = torch.tensor([pad(b["input_ids"], pad_id) for b in batch])
    attn = torch.tensor([pad(b["attention_mask"], 0) for b in batch])
    labels = torch.tensor([pad(b["labels"], -100) for b in batch])  # -100 = ignore in loss
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def train(model, tokenizer, train_ds, val_eval_pairs, cfg, accelerator=None):
    """Run the LoRA fine-tuning loop. Returns the best eval metrics seen.

    accelerator: Accelerator instance from the calling script. If None, a
    default single-process one is created (single-GPU / CPU fallback).
    """
    if accelerator is None:
        accelerator = Accelerator()

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer))
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)

    # accelerator.prepare() does three things at once:
    #   - wraps model in DistributedDataParallel and moves it to the right GPU
    #   - wraps optimizer so it steps across all replicas
    #   - wraps loader so each process gets a non-overlapping shard of the data
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    if accelerator.is_main_process:
        n = accelerator.num_processes
        log.info(f"training on {n} GPU(s), effective batch size = {cfg.batch_size * n}")

    best_em, stale, step = -1.0, 0, 0
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.train()
    # tqdm only on rank-0 to avoid N identical progress bars
    with tqdm(total=cfg.max_steps, desc="train", unit="step",
              disable=not accelerator.is_local_main_process) as progress:
        while step < cfg.max_steps:
            for batch in loader:
                # accelerate places tensors on the right device via prepare();
                # no manual .to(device) needed
                outputs = model(**batch)
                loss = outputs.loss
                # accelerator.backward handles gradient scaling (mixed-precision) and
                # the all-reduce across GPUs that DDP needs before optimizer.step()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if accelerator.is_local_main_process:
                    progress.update(1)
                    progress.set_postfix(loss=f"{loss.item():.4f}")
                if step % cfg.eval_every == 0:
                    if accelerator.is_main_process:
                        progress.set_postfix(loss=f"{loss.item():.4f}", phase="eval")
                        # unwrap_model strips DDP so .generate() works
                        raw = accelerator.unwrap_model(model)
                        metrics = evaluate(raw, tokenizer, val_eval_pairs)
                        model.train()
                        log.info(f"step {step} loss {loss.item():.4f} eval {metrics}")
                        if metrics["em"] > best_em:
                            best_em = metrics["em"]
                            stale = 0
                            raw.save_pretrained(os.path.join(cfg.output_dir, "best"))
                        else:
                            stale += 1
                        if stale >= cfg.early_stop_patience or step >= cfg.max_steps:
                            return {"best_em": best_em, "last": metrics}
                    # keep all ranks in sync after the eval window
                    accelerator.wait_for_everyone()
                if step >= cfg.max_steps:
                    break
    return {"best_em": best_em}
