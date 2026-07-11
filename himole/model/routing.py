"""Hierarchical sentence/token router scaffold for HiMoLE."""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class RoutingOutput:
    """Container for all routing tensors needed by the layer and losses."""

    # sentence_logits: scores over N KCGs from average-pooled sentence states.
    sentence_logits: torch.Tensor
    # token_logits: scores over N*M KCEs from each token state.
    token_logits: torch.Tensor
    # probabilities: softmax-normalized hierarchical expert probabilities.
    probabilities: torch.Tensor
    # topk_indices/topk_weights: sparse expert choices after KeepTop-k.
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor


class HierarchicalRouter(nn.Module):
    """Sentence-level KCG routing followed by token-level KCE refinement."""

    def __init__(self, hidden_size: int, num_kcgs: int, experts_per_kcg: int, top_k: int):
        super().__init__()
        if top_k <= 0 or top_k > num_kcgs * experts_per_kcg:
            raise ValueError("top_k must be between 1 and the total expert count")

        # This chunk implements the paper's W_sen linear router over KCGs.
        self.sentence_router = nn.Linear(hidden_size, num_kcgs, bias=False)

        # This chunk implements the paper's W_token linear router over all KCEs.
        self.token_router = nn.Linear(hidden_size, num_kcgs * experts_per_kcg, bias=False)

        # These topology values are needed to broadcast each sentence-level KCG
        # score across the M token-level expert scores in that KCG.
        self.num_kcgs = num_kcgs
        self.experts_per_kcg = experts_per_kcg
        self.top_k = top_k

    @property
    def total_experts(self) -> int:
        """Return N*M, the number of token-routed LoRA experts."""
        return self.num_kcgs * self.experts_per_kcg

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> RoutingOutput:
        """Compute G_sen, G_token, and sparse G_hie routing weights."""
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError("attention_mask must have shape [batch, sequence]")
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            h_sen = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        else:
            h_sen = hidden_states.mean(dim=1)
        G_sen = self.sentence_router(h_sen)
        G_token = self.token_router(hidden_states)
        G_sen_expanded = G_sen[:, None, :, None].expand(
            -1, hidden_states.shape[1], -1, self.experts_per_kcg
        ).reshape_as(G_token)
        G_hie = F.softmax(G_sen_expanded * G_token, dim=-1)
        topk_weights, topk_indices = G_hie.topk(self.top_k, dim=-1)
        # Re-normalize retained gates so an expert update keeps the scale of a
        # dense mixture while still dispatching gradients only to top-k experts.
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(G_hie.dtype).eps)
        return RoutingOutput(
            sentence_logits=G_sen,
            token_logits=G_token,
            probabilities=G_hie,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
        )
