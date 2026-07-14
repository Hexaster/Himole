"""Stage-1 KCG initialization for HiMoLE."""
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from himole.model.layer import HiMoLEFFNLayer
from himole.train.loop import accumulation_steps, collate, prepare_model_for_adapter_training

from himole.data.clustering import (
    assign_to_training_clusters,
    build_cluster_subsets,
    embed_training_examples,
    fit_training_clusters,
)


def _himole_layers(model):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, HiMoLEFFNLayer)]


def _trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _expert_state(layers, group_id):
    return {
        name: {
            key: value.detach().cpu().clone()
            for key, value in layer.kcgs[group_id].experts[0].state_dict().items()
        }
        for name, layer in layers
    }


def _load_expert_state(layers, group_id, state):
    for name, layer in layers:
        layer.kcgs[group_id].experts[0].load_state_dict(state[name], strict=True)


def apply_stage1_initializations(model, state_by_group):
    """Copy each trained KCG initialization into all experts in that group."""
    layers = _himole_layers(model)
    for name, layer in layers:
        for group_id, state_by_layer in state_by_group.items():
            layer.kcgs[group_id].load_stage1_initialization(state_by_layer[name])
    for parameter in model.parameters():
        parameter.requires_grad = False
    for _, layer in layers:
        for parameter in layer.router.parameters():
            parameter.requires_grad = True
        for group in layer.kcgs:
            for parameter in group.parameters():
                parameter.requires_grad = True
    return model


@torch.no_grad()
def _validation_loss(model, loader, device):
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        valid_tokens = int((batch["labels"] != -100).sum().item())
        if not valid_tokens:
            continue
        total_loss += model(**batch).loss.item() * valid_tokens
        total_tokens += valid_tokens
    model.train(was_training)
    return total_loss / total_tokens if total_tokens else float("inf")


def train_single_kcg(
    model,
    cluster_dataset,
    cfg,
    group_id: int,
    tokenizer=None,
    validation_dataset=None,
    progress_position=0,
):
    """Train one KCG independently on one semantic cluster."""
    if tokenizer is None:
        raise ValueError("tokenizer is required to collate Stage-1 examples")
    layers = _himole_layers(model)
    if not layers:
        raise ValueError("model must be patched with HiMoLE layers before Stage 1")
    if len(cluster_dataset) == 0:
        raise ValueError(f"cluster {group_id} is empty")

    for parameter in model.parameters():
        parameter.requires_grad = False
    for _, layer in layers:
        layer._stage1_group_id = group_id
        for parameter in layer.kcgs[group_id].experts[0].parameters():
            parameter.requires_grad = True

    prepare_model_for_adapter_training(model)
    device = next(model.parameters()).device
    loader = DataLoader(
        cluster_dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer),
    )
    validation_loader = None
    if validation_dataset is not None and len(validation_dataset):
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate(batch, tokenizer),
        )
    optimizer = AdamW(_trainable_parameters(model), lr=cfg.stage1_lr)
    max_steps = cfg.stage1_max_steps or cfg.max_steps
    accum = accumulation_steps(cfg)
    tqdm.write(
        f"Stage 1 KCG {group_id + 1}: {max_steps} optimizer steps, "
        f"micro-batch {cfg.micro_batch_size}, accumulation {accum}"
    )
    best_loss, stale, best_state = float("inf"), 0, None
    model.train()
    step, micro = 0, 0
    description = f"stage 1: KCG {group_id + 1}/{cfg.num_kcgs}"
    stopped_early = False
    with tqdm(total=max_steps, desc=description, unit="step", position=progress_position) as progress:
        while step < max_steps and not stopped_early:
            for batch in loader:
                batch = {name: value.to(device) for name, value in batch.items()}
                loss = model(**batch).loss / accum
                loss.backward()
                micro += 1
                if micro % accum:
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item() * accum:.4f}")
                if validation_loader is not None and step % cfg.eval_every == 0:
                    validation_loss = _validation_loss(model, validation_loader, device)
                    if validation_loss < best_loss:
                        best_loss = validation_loss
                        stale = 0
                        best_state = _expert_state(layers, group_id)
                    else:
                        stale += 1
                    progress.set_postfix(
                        loss=f"{loss.item() * accum:.4f}",
                        val_loss=f"{validation_loss:.4f}",
                        patience=f"{stale}/{cfg.early_stop_patience}",
                    )
                    stopped_early = stale >= cfg.early_stop_patience
                if step >= max_steps:
                    break
                if stopped_early:
                    tqdm.write(f"Stage 1 KCG {group_id + 1}: early stop at step {step}")
                    break

    if best_state is not None:
        _load_expert_state(layers, group_id, best_state)
    initialization = best_state or _expert_state(layers, group_id)
    for _, layer in layers:
        layer._stage1_group_id = None
    return initialization


def partition_stage1_datasets(
    train_dataset,
    validation_dataset,
    cfg,
    encoder=None,
    encoder_tokenizer=None,
):
    """Cluster training data and map validation data to the training centroids."""
    if validation_dataset is None:
        train_embeddings = embed_training_examples(
            train_dataset,
            cfg,
            encoder=encoder,
            tokenizer=encoder_tokenizer,
        )
        train_ids, _ = fit_training_clusters(train_embeddings, cfg)
        return build_cluster_subsets(train_dataset, train_ids.tolist(), cfg), None

    owns_encoder = encoder is None
    if encoder is None or encoder_tokenizer is None:
        from transformers import AutoModel, AutoTokenizer

        encoder_tokenizer = encoder_tokenizer or AutoTokenizer.from_pretrained(cfg.clustering_encoder)
        encoder = encoder or AutoModel.from_pretrained(cfg.clustering_encoder)
    if owns_encoder and torch.cuda.is_available():
        encoder.to("cuda")

    try:
        train_embeddings = embed_training_examples(
            train_dataset,
            cfg,
            encoder=encoder,
            tokenizer=encoder_tokenizer,
        )
        train_ids, centers = fit_training_clusters(train_embeddings, cfg)
        train_subsets = build_cluster_subsets(train_dataset, train_ids.tolist(), cfg)
        validation_subsets = None
        if validation_dataset is not None:
            validation_embeddings = embed_training_examples(
                validation_dataset,
                cfg,
                encoder=encoder,
                tokenizer=encoder_tokenizer,
            )
            validation_ids = assign_to_training_clusters(validation_embeddings, centers)
            validation_subsets = build_cluster_subsets(
                validation_dataset,
                validation_ids.tolist(),
                cfg,
            )
    finally:
        if owns_encoder:
            del encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return train_subsets, validation_subsets


def _checkpoint_signature(cfg, group_id, train_size, validation_size):
    return {
        "group_id": group_id,
        "base_model": cfg.base_model,
        "clustering_encoder": cfg.clustering_encoder,
        "id_dataset": cfg.id_dataset,
        "cutoff_len": cfg.cutoff_len,
        "max_train_samples": cfg.max_train_samples,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "num_kcgs": cfg.num_kcgs,
        "target_modules": tuple(cfg.target_modules),
        "stage1_lr": cfg.stage1_lr,
        "stage1_max_steps": cfg.stage1_max_steps or cfg.max_steps,
        "batch_size": cfg.batch_size,
        "micro_batch_size": cfg.micro_batch_size,
        "eval_every": cfg.eval_every,
        "early_stop_patience": cfg.early_stop_patience,
        "train_size": train_size,
        "validation_size": validation_size,
        "seed": cfg.seed,
    }


def _load_group_checkpoint(path, expected_signature):
    path = Path(path)
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("signature") != expected_signature:
        return None
    return payload["state"]


def _train_kcg_worker(
    local_rank,
    group_ids,
    train_subsets,
    validation_subsets,
    cfg,
    checkpoint_paths,
    signatures,
):
    """Train one KCG in one spawned process on one CUDA device."""
    from himole.model.baseline_lora import load_base_model, load_tokenizer
    from himole.model.patching import attach_himole_to_model
    from himole.utils import set_seed

    group_id = group_ids[local_rank]
    torch.cuda.set_device(local_rank)
    set_seed(cfg.seed + group_id)
    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg).to(f"cuda:{local_rank}")
    model = attach_himole_to_model(model, cfg)
    state = train_single_kcg(
        model,
        train_subsets[local_rank],
        cfg,
        group_id,
        tokenizer=tokenizer,
        validation_dataset=validation_subsets[local_rank],
        progress_position=local_rank,
    )

    checkpoint_path = Path(checkpoint_paths[local_rank])
    temporary_path = checkpoint_path.with_suffix(f".{os.getpid()}.tmp")
    torch.save({"signature": signatures[local_rank], "state": state}, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def initialize_kcgs_parallel(
    train_dataset,
    validation_dataset,
    cfg,
    resume=False,
    launcher=None,
):
    """Train KCGs concurrently, using one spawned process and GPU per KCG."""
    cfg.validate()
    if validation_dataset is None:
        raise ValueError("validation_dataset is required for parallel Stage 1 early stopping")
    if torch.cuda.device_count() < cfg.num_kcgs:
        raise RuntimeError(f"parallel Stage 1 requires {cfg.num_kcgs} visible CUDA devices")

    train_subsets, validation_subsets = partition_stage1_datasets(
        train_dataset,
        validation_dataset,
        cfg,
    )
    if any(not subset for subset in train_subsets):
        raise RuntimeError("K-means produced an empty training cluster")
    if any(not subset for subset in validation_subsets):
        raise RuntimeError("a KCG has no validation examples; increase the validation sample count")

    output_dir = Path(cfg.stage1_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = [output_dir / f"kcg_{group_id}.pt" for group_id in range(cfg.num_kcgs)]
    signatures = [
        _checkpoint_signature(cfg, group_id, len(train_subsets[group_id]), len(validation_subsets[group_id]))
        for group_id in range(cfg.num_kcgs)
    ]
    state_by_group = {}
    if resume:
        for group_id, (path, signature) in enumerate(zip(checkpoint_paths, signatures)):
            state = _load_group_checkpoint(path, signature)
            if state is not None:
                state_by_group[group_id] = state
                tqdm.write(f"Stage 1 KCG {group_id + 1}: resumed completed checkpoint")

    pending = [group_id for group_id in range(cfg.num_kcgs) if group_id not in state_by_group]
    if pending:
        pending_train = [train_subsets[group_id] for group_id in pending]
        pending_validation = [validation_subsets[group_id] for group_id in pending]
        pending_paths = [str(checkpoint_paths[group_id]) for group_id in pending]
        pending_signatures = [signatures[group_id] for group_id in pending]
        if launcher is None:
            from torch.multiprocessing import spawn

            launcher = spawn
        launcher(
            _train_kcg_worker,
            args=(
                pending,
                pending_train,
                pending_validation,
                cfg,
                pending_paths,
                pending_signatures,
            ),
            nprocs=len(pending),
            join=True,
        )
        for group_id in pending:
            state = _load_group_checkpoint(checkpoint_paths[group_id], signatures[group_id])
            if state is None:
                raise RuntimeError(f"Stage 1 worker did not produce a valid KCG {group_id + 1} checkpoint")
            state_by_group[group_id] = state
    return state_by_group


def initialize_kcgs(
    model,
    train_dataset,
    cfg,
    tokenizer=None,
    encoder=None,
    encoder_tokenizer=None,
    validation_dataset=None,
):
    """Embed, cluster, and independently train all N KCG initializations."""
    cfg.validate()
    subsets, validation_subsets = partition_stage1_datasets(
        train_dataset,
        validation_dataset,
        cfg,
        encoder=encoder,
        encoder_tokenizer=encoder_tokenizer,
    )
    tqdm.write(f"Stage 1: clustered {len(train_dataset)} examples into {cfg.num_kcgs} KCGs")
    if any(not subset for subset in subsets):
        raise RuntimeError("K-means produced an empty cluster; reduce num_kcgs or add data")

    state_by_group = {}
    for group_id, subset in enumerate(subsets):
        group_validation = validation_subsets[group_id] if validation_subsets is not None else None
        state_by_group[group_id] = train_single_kcg(
            model,
            subset,
            cfg,
            group_id,
            tokenizer=tokenizer,
            validation_dataset=group_validation,
        )

    apply_stage1_initializations(model, state_by_group)
    return state_by_group


__all__ = [
    "apply_stage1_initializations",
    "initialize_kcgs",
    "initialize_kcgs_parallel",
    "partition_stage1_datasets",
    "train_single_kcg",
]
