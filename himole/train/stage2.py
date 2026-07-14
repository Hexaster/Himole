"""Stage-2 joint optimization of HiMoLE experts and routers."""
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from himole.model.layer import HiMoLEFFNLayer
from himole.train.loop import accumulation_steps, collate, prepare_model_for_adapter_training
from himole.train.losses import auxiliary_load_balancing_loss, combine_stage2_losses, diversity_loss


def _stage2_state(layers):
    return {
        str(index): {
            "router": {
                key: value.detach().cpu().clone()
                for key, value in layer.router.state_dict().items()
            },
            "kcgs": {
                key: value.detach().cpu().clone()
                for key, value in layer.kcgs.state_dict().items()
            },
        }
        for index, layer in enumerate(layers)
    }


def _load_stage2_state(layers, state):
    for index, layer in enumerate(layers):
        layer.router.load_state_dict(state[str(index)]["router"], strict=True)
        layer.kcgs.load_state_dict(state[str(index)]["kcgs"], strict=True)


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
    best_state = None
    accum = accumulation_steps(cfg)
    model.train()
    with tqdm(total=cfg.max_steps, desc="stage 2", unit="step") as progress:
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
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item() * accum:.4f}")

                # These tensors retain the most recent autograd graph. They are only
                # diagnostics for the current step and must not keep activations alive.
                for layer in layers:
                    layer.last_routing = None
                    layer.last_kcg_outputs = None

                if val_eval_pairs and step % cfg.eval_every == 0:
                    from himole.eval.generate import evaluate

                    progress.set_postfix(loss=f"{loss.item() * accum:.4f}", phase="eval")
                    metrics = evaluate(
                        model,
                        tokenizer,
                        val_eval_pairs,
                        cutoff_len=cfg.cutoff_len,
                        batch_size=cfg.eval_batch_size,
                    )
                    model.train()
                    for layer in layers:
                        layer.last_routing = None
                        layer.last_kcg_outputs = None
                    if metrics["em"] > best_em:
                        best_em, stale = metrics["em"], 0
                        best_state = _stage2_state(layers)
                        checkpoint_path = os.path.join(cfg.stage2_output_dir, "best.pt")
                        temporary_path = f"{checkpoint_path}.tmp"
                        torch.save(best_state, temporary_path)
                        os.replace(temporary_path, checkpoint_path)
                    else:
                        stale += 1
                    if stale >= cfg.early_stop_patience:
                        if best_state is not None:
                            _load_stage2_state(layers, best_state)
                        return {"best_em": best_em, "steps": step}
                if step >= cfg.max_steps:
                    break
    if best_state is not None:
        _load_stage2_state(layers, best_state)
    return {"best_em": best_em, "steps": step}
