"""LoRA expert and KCG containers for the strict HiMoLE scaffold."""
import math

import torch
from torch import nn


class LoRAExpert(nn.Module):
    """One Knowledge Collaboration Expert (KCE), parameterized as E = B*A."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: int, dropout: float):
        super().__init__()
        # This chunk stores the low-rank A and B matrices using PyTorch Linear
        # modules. Their weight tensors are transposed relative to the paper's
        # matrix notation, but the logical mapping is h -> A -> B.
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize A randomly and B at zero, matching common LoRA practice."""
        # This chunk gives each expert a harmless initial delta: A can learn a
        # projection, while zero B makes the adapter contribute nothing at start.
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the low-rank expert update to token hidden states."""
        return self.scaling * self.lora_b(self.dropout(self.lora_a(hidden_states)))


class KnowledgeCompetitionGroup(nn.Module):
    """A KCG: M LoRA experts that share the same Stage-1 initialization."""

    def __init__(
        self,
        group_id: int,
        num_experts: int,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int,
        dropout: float,
    ):
        super().__init__()
        # This chunk records which semantic cluster this group represents after
        # Stage 1. The id is structural metadata, not a learned parameter.
        self.group_id = group_id

        # This chunk creates the M KCEs inside the group. Stage 1 should later
        # copy the independently trained KCG initialization into all M experts.
        self.experts = nn.ModuleList(
            LoRAExpert(in_features, out_features, rank, alpha, dropout)
            for _ in range(num_experts)
        )

    def load_stage1_initialization(self, state_dict) -> None:
        """Copy a Stage-1-trained KCG initialization into every KCE in the group."""
        if "lora_a.weight" in state_dict:
            source = state_dict
        else:
            source = {
                "lora_a.weight": state_dict["experts.0.lora_a.weight"],
                "lora_b.weight": state_dict["experts.0.lora_b.weight"],
            }
        for expert in self.experts:
            expert.load_state_dict(source, strict=True)


    def forward(self, hidden_states: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        """Combine the M expert outputs inside this KCG for one forward pass."""
        if expert_weights.shape[:-1] != hidden_states.shape[:-1]:
            raise ValueError("expert_weights must align with hidden_states batch and sequence dimensions")
        if expert_weights.shape[-1] != len(self.experts):
            raise ValueError("expert_weights has the wrong number of experts")
        expert_outputs = []

        # 1. Run every LoRAExpert on the same hidden states.
        for expert in self.experts:
            expert_outputs.append(expert(hidden_states))
        # 2. Stack outputs into shape [batch, sequence, num_experts, hidden].
        stacked_outputs = torch.stack(expert_outputs, dim=-2)

        # 3. Accept router weights and compute the weighted expert sum.
        weighted_outputs = stacked_outputs * expert_weights.unsqueeze(-1)
        combined_output = weighted_outputs.sum(dim=-2)
        return combined_output
