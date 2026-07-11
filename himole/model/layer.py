"""HiMoLE replacement layer scaffold for transformer FFN modules."""
import torch
from torch import nn

from himole.model.experts import KnowledgeCompetitionGroup
from himole.model.routing import HierarchicalRouter


class HiMoLEFFNLayer(nn.Module):
    """Wrap a frozen FFN module with hierarchical LoRA expert updates."""

    def __init__(self, base_ffn: nn.Module, hidden_size: int, out_features: int, cfg):
        super().__init__()
        cfg.validate()

        # This chunk keeps the original FFN path W0*h_token. It should remain
        # frozen, because HiMoLE is a parameter-efficient adaptation method.
        self.base_ffn = base_ffn
        for param in self.base_ffn.parameters():
            param.requires_grad = False

        # This chunk owns the hierarchical sentence/token routing module.
        self.router = HierarchicalRouter(
            hidden_size=hidden_size,
            num_kcgs=cfg.num_kcgs,
            experts_per_kcg=cfg.experts_per_kcg,
            top_k=cfg.top_k,
        )

        # This chunk creates the N KCGs, each containing M KCE LoRA experts.
        self.kcgs = nn.ModuleList(
            KnowledgeCompetitionGroup(
                group_id=group_id,
                num_experts=cfg.experts_per_kcg,
                in_features=hidden_size,
                out_features=out_features,
                rank=cfg.lora_r,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
            )
            for group_id in range(cfg.num_kcgs)
        )
        self.last_routing = None
        self.last_kcg_outputs = None
        self._stage1_group_id = None

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return W0*h plus the routed sum of LoRA expert outputs."""
        base_output = self.base_ffn(hidden_states)
        if self._stage1_group_id is not None:
            output = self.kcgs[self._stage1_group_id].experts[0](hidden_states)
            self.last_routing = None
            self.last_kcg_outputs = None
            return base_output + output

        routing = self.router(hidden_states, attention_mask)
        gates = torch.zeros_like(routing.probabilities).scatter_(-1, routing.topk_indices, routing.topk_weights)
        grouped_gates = gates.view(*gates.shape[:-1], len(self.kcgs), -1)
        kcg_outputs = [
            group(hidden_states, grouped_gates[..., group_id, :])
            for group_id, group in enumerate(self.kcgs)
        ]
        self.last_routing = routing
        self.last_kcg_outputs = kcg_outputs
        return base_output + torch.stack(kcg_outputs, dim=0).sum(dim=0)
