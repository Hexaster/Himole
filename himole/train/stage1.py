"""Stage-1 KCG initialization for HiMoLE."""
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from himole.model.layer import HiMoLEFFNLayer
from himole.train.loop import accumulation_steps, collate, prepare_model_for_adapter_training

from himole.data.clustering import build_cluster_subsets, cluster_training_examples, embed_training_examples


def _himole_layers(model):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, HiMoLEFFNLayer)]


def _trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def train_single_kcg(model, cluster_dataset, cfg, group_id: int, tokenizer=None):
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
    optimizer = AdamW(_trainable_parameters(model), lr=cfg.stage1_lr)
    max_steps = cfg.stage1_max_steps or cfg.max_steps
    accum = accumulation_steps(cfg)
    model.train()
    step, micro = 0, 0
    description = f"stage 1: KCG {group_id + 1}/{cfg.num_kcgs}"
    with tqdm(total=max_steps, desc=description, unit="step") as progress:
        while step < max_steps:
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
                if step >= max_steps:
                    break

    initialization = {}
    for name, layer in layers:
        expert = layer.kcgs[group_id].experts[0]
        initialization[name] = {key: value.detach().cpu().clone() for key, value in expert.state_dict().items()}
        layer._stage1_group_id = None
    return initialization


def initialize_kcgs(model, train_dataset, cfg, tokenizer=None, encoder=None, encoder_tokenizer=None):
    """Embed, cluster, and independently train all N KCG initializations."""
    cfg.validate()
    examples = list(train_dataset)
    embeddings = embed_training_examples(examples, cfg, encoder=encoder, tokenizer=encoder_tokenizer)
    tqdm.write(f"Stage 1: clustering {len(examples)} examples into {cfg.num_kcgs} KCGs")
    cluster_ids = cluster_training_examples(embeddings, cfg)
    subsets = build_cluster_subsets(examples, cluster_ids.tolist(), cfg)
    if any(not subset for subset in subsets):
        raise RuntimeError("K-means produced an empty cluster; reduce num_kcgs or add data")

    state_by_group = {}
    for group_id, subset in enumerate(subsets):
        state_by_group[group_id] = train_single_kcg(model, subset, cfg, group_id, tokenizer=tokenizer)

    for name, layer in _himole_layers(model):
        for group_id, state_by_layer in state_by_group.items():
            layer.kcgs[group_id].load_stage1_initialization(state_by_layer[name])
    for parameter in model.parameters():
        parameter.requires_grad = False
    for _, layer in _himole_layers(model):
        for parameter in layer.base_ffn.parameters():
            parameter.requires_grad = False
        for parameter in layer.router.parameters():
            parameter.requires_grad = True
        for group in layer.kcgs:
            for parameter in group.parameters():
                parameter.requires_grad = True
    return state_by_group


__all__ = ["cluster_training_examples", "embed_training_examples", "initialize_kcgs", "train_single_kcg"]
