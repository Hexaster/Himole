"""Behavior tests for HiMoLE experts, routing, layers, and model patching."""

import torch

from himole.config import HimoleConfig
from himole.model.experts import LoRAExpert
from himole.model.layer import HiMoLEFFNLayer
from himole.model.patching import attach_himole_to_model
from himole.model.routing import HierarchicalRouter


def _small_config(**overrides):
    values = {
        "num_kcgs": 2,
        "experts_per_kcg": 2,
        "top_k": 1,
        "lora_r": 1,
        "lora_alpha": 2,
        "lora_dropout": 0.0,
    }
    values.update(overrides)
    return HimoleConfig(**values)


def test_lora_expert_applies_low_rank_update_and_scaling():
    expert = LoRAExpert(in_features=2, out_features=1, rank=1, alpha=2, dropout=0.0)
    with torch.no_grad():
        expert.lora_a.weight.copy_(torch.tensor([[1.0, 2.0]]))
        expert.lora_b.weight.fill_(3.0)

    output = expert(torch.tensor([[[2.0, 1.0]]]))

    # (2*1 + 1*2) * 3 * (alpha/rank)
    assert torch.equal(output, torch.tensor([[[24.0]]]))


def test_router_mask_excludes_padding_from_sentence_pooling():
    router = HierarchicalRouter(hidden_size=2, num_kcgs=2, experts_per_kcg=1, top_k=1)
    hidden_states = torch.tensor([[[1.0, 3.0], [99.0, 99.0]]])

    masked = router(hidden_states, attention_mask=torch.tensor([[1, 0]]))
    unpadded = router(hidden_states[:, :1])

    assert torch.allclose(masked.sentence_logits, unpadded.sentence_logits)


def test_router_keeps_top_k_independently_for_each_token():
    router = HierarchicalRouter(hidden_size=2, num_kcgs=2, experts_per_kcg=2, top_k=2)
    output = router(torch.randn(2, 3, 2))
    sparse_gates = torch.zeros_like(output.probabilities).scatter_(
        -1, output.topk_indices, output.topk_weights
    )

    assert output.probabilities.shape == (2, 3, 4)
    assert output.topk_indices.shape == (2, 3, 2)
    assert torch.equal((sparse_gates != 0).sum(dim=-1), torch.full((2, 3), 2))
    assert torch.allclose(sparse_gates.sum(dim=-1), torch.ones(2, 3))


def test_router_rejects_invalid_top_k_and_mask_shape():
    for top_k in (0, 5):
        try:
            HierarchicalRouter(2, num_kcgs=2, experts_per_kcg=2, top_k=top_k)
        except ValueError as error:
            assert "top_k" in str(error)
        else:
            raise AssertionError("invalid top_k was accepted")

    router = HierarchicalRouter(2, num_kcgs=1, experts_per_kcg=1, top_k=1)
    try:
        router(torch.randn(2, 3, 2), attention_mask=torch.ones(2, 2))
    except ValueError as error:
        assert "attention_mask" in str(error)
    else:
        raise AssertionError("misaligned attention mask was accepted")


def test_himole_layer_freezes_base_and_backpropagates_to_experts_and_router():
    base = torch.nn.Linear(2, 2)
    layer = HiMoLEFFNLayer(base, hidden_size=2, out_features=2, cfg=_small_config())
    for group in layer.kcgs:
        for expert in group.experts:
            torch.nn.init.ones_(expert.lora_b.weight)

    layer(torch.randn(2, 3, 2)).sum().backward()

    assert all(parameter.grad is None for parameter in base.parameters())
    assert any(
        expert.lora_a.weight.grad is not None
        for group in layer.kcgs
        for expert in group.experts
    )
    assert all(parameter.grad is not None for parameter in layer.router.parameters())


def test_patching_leaves_only_himole_parameters_trainable():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 2)
            self.mlp = torch.nn.Module()
            self.mlp.c_fc = torch.nn.Linear(2, 4)
            self.mlp.c_proj = torch.nn.Linear(4, 2)
            self.head = torch.nn.Linear(2, 8)

    model = attach_himole_to_model(ToyModel(), _small_config(use_tiny=True))
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    assert trainable_names
    assert all("router" in name or "experts" in name for name in trainable_names)
    assert not model.embedding.weight.requires_grad
    assert not model.head.weight.requires_grad
