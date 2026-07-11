"""Stage-1 sentence embedding and deterministic K-means clustering.

HiMoLE first embeds each training sentence, clusters those embeddings into N
subsets, and trains one KCG per cluster before joint routing is learned.
"""
from collections.abc import Mapping

import torch


def _example_text(example) -> str:
    """Extract a semantic sentence from the dataset shapes used by this project."""
    if isinstance(example, str):
        return example
    if not isinstance(example, Mapping):
        raise TypeError("examples must be strings or mappings")
    if "text" in example:
        return str(example["text"])
    if "context" in example:
        question = example.get("question", "")
        return f"{example['context']}\n{question}".strip()
    if "prompt" in example:
        return str(example["prompt"])
    if "clustering_text" in example:
        return str(example["clustering_text"])
    raise KeyError("example needs a text, context, or prompt field")


def embed_training_examples(examples, cfg, encoder=None, tokenizer=None):
    """Return one semantic embedding per training example for Stage 1 clustering."""
    cfg.validate()
    examples = list(examples)
    if not examples:
        return torch.empty((0, 0), dtype=torch.float32)

    owns_encoder = encoder is None
    if encoder is None or tokenizer is None:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.clustering_encoder)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoder = encoder or AutoModel.from_pretrained(cfg.clustering_encoder)

    if owns_encoder and torch.cuda.is_available():
        encoder.to("cuda")
    device = next(encoder.parameters()).device
    was_training = encoder.training
    encoder.eval()
    embeddings = []
    texts = [_example_text(example) for example in examples]
    try:
        with torch.no_grad():
            for start in range(0, len(texts), cfg.clustering_batch_size):
                batch = tokenizer(
                    texts[start:start + cfg.clustering_batch_size],
                    padding=True,
                    truncation=True,
                    max_length=cfg.cutoff_len,
                    return_tensors="pt",
                )
                batch = {name: value.to(device) for name, value in batch.items()}
                outputs = encoder(**batch)
                states = outputs.last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(states.dtype)
                pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                embeddings.append(pooled.cpu())
    finally:
        encoder.train(was_training)
    result = torch.cat(embeddings, dim=0)
    if owns_encoder:
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def cluster_training_examples(embeddings, cfg):
    """Assign each embedded example to one of cfg.num_kcgs K-means clusters."""
    cfg.validate()
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_examples, hidden_size]")
    if len(embeddings) < cfg.num_kcgs:
        raise ValueError("need at least num_kcgs examples to initialize K-means")

    # Evenly spaced starting points make the split reproducible without adding a
    # scikit-learn dependency to the small training environment.
    initial = torch.linspace(0, len(embeddings) - 1, cfg.num_kcgs).round().long()
    centers = embeddings[initial].clone()
    for _ in range(cfg.clustering_max_iterations):
        distances = torch.cdist(embeddings, centers)
        cluster_ids = distances.argmin(dim=1)
        next_centers = centers.clone()
        for cluster_id in range(cfg.num_kcgs):
            members = embeddings[cluster_ids == cluster_id]
            if len(members):
                next_centers[cluster_id] = members.mean(dim=0)
        if torch.allclose(next_centers, centers):
            break
        centers = next_centers
    return cluster_ids


def build_cluster_subsets(examples, cluster_ids, cfg):
    """Return N example lists, one per KCG, after clustering."""
    # This utility is intentionally simple: once cluster_ids are known, Stage 1
    # needs ordinary per-cluster sub-datasets for independent KCG initialization.
    cfg.validate()
    if len(examples) != len(cluster_ids):
        raise ValueError("examples and cluster_ids must have the same length")
    subsets = [[] for _ in range(cfg.num_kcgs)]
    for example, cluster_id in zip(examples, cluster_ids):
        if not 0 <= int(cluster_id) < cfg.num_kcgs:
            raise ValueError("cluster_ids must be between 0 and num_kcgs - 1")
        subsets[int(cluster_id)].append(example)
    return subsets
