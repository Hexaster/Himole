"""A transparent, hand-written training loop so every step of LoRA fine-tuning
is visible: build batches -> forward -> loss -> backward -> optimizer step,
with periodic eval, early stopping, and best-checkpoint saving.
"""
import os

import torch
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


def train(model, tokenizer, train_ds, val_eval_pairs, cfg):
    """Run the LoRA fine-tuning loop. Returns the best eval metrics seen."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if device == "cuda":
        log.info(f"training on cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name()})")
    else:
        log.info("training on cpu")

    # Memory: recompute activations during backward instead of storing all of them.
    # Without this, a 7B fwd/bwd over a full micro-batch of 1024-token sequences OOMs
    # even on a 40GB A100 (weights ~14GB fit; the retained activations are what overflow).
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # frozen base: let grads reach the LoRA adapters
    model.config.use_cache = False      # KV cache is incompatible with checkpointing while training

    # Reach the paper's effective batch (cfg.batch_size) by accumulating grads over
    # cfg.batch_size / cfg.micro_batch_size forward passes before each optimizer step.
    accum = max(1, cfg.batch_size // cfg.micro_batch_size)
    loader = DataLoader(train_ds, batch_size=cfg.micro_batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer))
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)

    best_em, stale, step, micro = -1.0, 0, 0, 0
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.train()
    with tqdm(total=cfg.max_steps, desc="train", unit="step") as progress:
        while step < cfg.max_steps:
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                # Forward/backward on ONE micro-batch. Scale the loss by 1/accum so the
                # accumulated grads average (not sum) to the effective-batch gradient.
                outputs = model(**batch)          # HF returns outputs.loss (CE on labels)
                loss = outputs.loss / accum
                loss.backward()
                micro += 1
                if micro % accum != 0:
                    continue                       # keep accumulating; no optimizer step yet
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item() * accum:.4f}")
                if step % cfg.eval_every == 0:                 # periodic eval + early stop
                    progress.set_postfix(loss=f"{loss.item() * accum:.4f}", phase="eval")
                    metrics = evaluate(model, tokenizer, val_eval_pairs, cutoff_len=cfg.cutoff_len)
                    model.train()
                    log.info(f"step {step} loss {loss.item() * accum:.4f} eval {metrics}")
                    if metrics["em"] > best_em:
                        best_em = metrics["em"]
                        stale = 0
                        model.save_pretrained(os.path.join(cfg.output_dir, "best"))
                    else:
                        stale += 1
                    if stale >= cfg.early_stop_patience or step >= cfg.max_steps:
                        return {"best_em": best_em, "last": metrics}
                if step >= cfg.max_steps:
                    break
    return {"best_em": best_em}
