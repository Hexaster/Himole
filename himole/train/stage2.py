"""Stage-2 joint optimization of HiMoLE experts and routers."""
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from himole.eval.generate import evaluate
from himole.model.layer import HiMoLEFFNLayer
from himole.train.loop import accumulation_steps, collate, prepare_model_for_adapter_training
from himole.train.losses import auxiliary_load_balancing_loss, combine_stage2_losses, diversity_loss


def train_himole_stage2(model, tokenizer, train_ds, cfg, val_eval_pairs=None):
    """Jointly optimize KCE experts and hierarchical routers on the full dataset."""
    cfg.validate()
    if tokenizer is None:
        raise ValueError("tokenizer is required to collate Stage-2 examples")
    layers = [module for module in model.modules() if isinstance(module, HiMoLEFFNLayer)]
    if not layers:
        raise ValueError("model must be patched with HiMoLE layers before Stage 2")
    if len(train_ds) == 0:
        raise ValueError("train_ds must not be empty")

    device = next(model.parameters()).device
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in layers:
        for parameter in layer.router.parameters():
            parameter.requires_grad = True
        for group in layer.kcgs:
            for parameter in group.parameters():
                parameter.requires_grad = True

    prepare_model_for_adapter_training(model)

    loader = DataLoader(train_ds, batch_size=cfg.micro_batch_size, shuffle=True, collate_fn=lambda batch: collate(batch, tokenizer))
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=cfg.stage2_lr)
    os.makedirs(cfg.stage2_output_dir, exist_ok=True)
    best_em, stale, step, micro = -1.0, 0, 0, 0
    accum = accumulation_steps(cfg)
    model.train()
    while step < cfg.max_steps:
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            outputs = model(**batch)
            aux = torch.stack([
                auxiliary_load_balancing_loss(layer.last_routing.probabilities, layer.last_routing.topk_indices, cfg.total_experts)
                for layer in layers
            ]).mean()
            diverse_layers = layers[::cfg.diversity_every_n_layers]
            diverse = torch.stack([diversity_loss(layer.last_kcg_outputs, cfg.diversity_eps) for layer in diverse_layers]).mean()
            loss = combine_stage2_losses(outputs.loss, aux, diverse, cfg) / accum
            loss.backward()
            micro += 1
            if micro % accum:
                continue
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # These tensors retain the most recent autograd graph. They are only
            # diagnostics for the current step and must not keep activations alive.
            for layer in layers:
                layer.last_routing = None
                layer.last_kcg_outputs = None

            if val_eval_pairs and step % cfg.eval_every == 0:
                metrics = evaluate(
                    model,
                    tokenizer,
                    val_eval_pairs,
                    cutoff_len=cfg.cutoff_len,
                    batch_size=cfg.eval_batch_size,
                )
                model.train()
                if metrics["em"] > best_em:
                    best_em, stale = metrics["em"], 0
                    model.save_pretrained(os.path.join(cfg.stage2_output_dir, "best"))
                else:
                    stale += 1
                if stale >= cfg.early_stop_patience:
                    return {"best_em": best_em, "steps": step}
            if step >= cfg.max_steps:
                break
    return {"best_em": best_em, "steps": step}
