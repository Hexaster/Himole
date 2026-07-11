"""Unit tests for the implemented HiMoLE building blocks."""
from types import SimpleNamespace

import torch

from himole.config import HimoleConfig
from himole.data.clustering import build_cluster_subsets, cluster_training_examples
from himole.eval.load_balance import maxvio_global
from himole.model.experts import KnowledgeCompetitionGroup, LoRAExpert
from himole.model.layer import HiMoLEFFNLayer
from himole.model.patching import attach_himole_to_model
from himole.model.routing import HierarchicalRouter
from himole.train.losses import auxiliary_load_balancing_loss, combine_stage2_losses, diversity_loss
from himole.train.stage1 import train_single_kcg
from himole.train.stage2 import train_himole_stage2


class _Tokenizer:
    pad_token_id = 0


class _TinyCausalModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.gradient_checkpointing_enabled = False
        self.input_grads_enabled = False
        self.embedding = torch.nn.Embedding(8, 4)
        self.projection = HiMoLEFFNLayer(torch.nn.Linear(4, 4), 4, 4, cfg)
        self.head = torch.nn.Linear(4, 8)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.head(self.projection(self.embedding(input_ids), attention_mask))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 8), labels.reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def enable_input_require_grads(self):
        self.input_grads_enabled = True


_DATASET = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2], "clustering_text": "x"}]


def test_himole_config_defaults_match_eqa_build():
    cfg = HimoleConfig()

    assert cfg.num_kcgs == 3
    assert cfg.experts_per_kcg == 4
    assert cfg.total_experts == 12
    assert cfg.lora_r == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.05
    assert cfg.top_k == 2
    assert cfg.stage1_lr == 3e-4
    assert cfg.stage2_lr == 3e-5
    assert cfg.target_modules == ("up_proj", "down_proj", "gate_proj")
    assert cfg.clustering_encoder == "microsoft/deberta-v3-large"


def test_lora_expert_and_group_hold_paper_shapes():
    expert = LoRAExpert(in_features=8, out_features=6, rank=2, alpha=4, dropout=0.05)
    group = KnowledgeCompetitionGroup(1, 4, 8, 6, 2, 4, 0.05)

    assert expert.lora_a.in_features == 8
    assert expert.lora_a.out_features == 2
    assert expert.lora_b.in_features == 2
    assert expert.lora_b.out_features == 6
    assert expert.scaling == 2.0
    assert len(group.experts) == 4


def test_kcg_forward_combines_weighted_experts_and_propagates_gradients():
    group = KnowledgeCompetitionGroup(0, 3, 4, 4, 2, 2, 0.0)
    hidden_states = torch.randn(2, 3, 4)
    expert_weights = torch.zeros(2, 3, 3)
    expert_weights[..., 1] = 1.0
    for expert in group.experts:
        torch.nn.init.ones_(expert.lora_b.weight)

    output = group(hidden_states, expert_weights)
    assert output.shape == (2, 3, 4)
    assert torch.allclose(output, group.experts[1](hidden_states))
    output.sum().backward()
    assert group.experts[1].lora_a.weight.grad is not None
    assert group.experts[1].lora_b.weight.grad is not None


def test_router_returns_normalized_sparse_topk_weights():
    router = HierarchicalRouter(hidden_size=8, num_kcgs=3, experts_per_kcg=4, top_k=2)
    output = router(torch.zeros(2, 5, 8), attention_mask=torch.ones(2, 5))

    assert output.sentence_logits.shape == (2, 3)
    assert output.token_logits.shape == (2, 5, 12)
    assert output.probabilities.shape == (2, 5, 12)
    assert output.topk_indices.shape == (2, 5, 2)
    assert torch.allclose(output.topk_weights.sum(dim=-1), torch.ones(2, 5))


def test_himole_layer_adds_sparse_expert_updates_to_frozen_base_ffn():
    cfg = HimoleConfig(num_kcgs=2, experts_per_kcg=2, top_k=1, lora_r=2, lora_alpha=2)
    base_ffn = torch.nn.Linear(4, 6, bias=False)
    layer = HiMoLEFFNLayer(base_ffn=base_ffn, hidden_size=4, out_features=6, cfg=cfg)
    for group in layer.kcgs:
        for expert in group.experts:
            torch.nn.init.ones_(expert.lora_b.weight)
    result = layer(torch.randn(2, 5, 4))

    assert result.shape == (2, 5, 6)
    assert layer.last_routing.topk_indices.shape == (2, 5, 1)
    assert len(layer.last_kcg_outputs) == 2
    assert not base_ffn.weight.requires_grad


def test_patching_replaces_only_gpt_mlp_projections():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.c_fc = torch.nn.Linear(4, 8)
            self.mlp.c_proj = torch.nn.Linear(8, 4)
            self.attn = torch.nn.Module()
            self.attn.c_proj = torch.nn.Linear(4, 4)

    cfg = HimoleConfig(use_tiny=True, num_kcgs=1, experts_per_kcg=2, top_k=1, lora_r=2)
    model = attach_himole_to_model(ToyModel(), cfg)

    assert isinstance(model.mlp.c_fc, HiMoLEFFNLayer)
    assert isinstance(model.mlp.c_proj, HiMoLEFFNLayer)
    assert isinstance(model.attn.c_proj, torch.nn.Linear)


def test_losses_and_load_balance_match_small_hand_computable_cases():
    probabilities = torch.tensor([[[0.8, 0.2], [0.7, 0.3]]])
    topk_indices = torch.tensor([[[0], [0]]])
    # F=[1,0], P=[.75,.25], so L_aux=2*.75.
    assert torch.allclose(auxiliary_load_balancing_loss(probabilities, topk_indices, 2), torch.tensor(1.5))

    same = torch.ones(1, 1, 2)
    opposite = -torch.ones(1, 1, 2)
    assert torch.allclose(diversity_loss([same, opposite]), torch.tensor(-1.0))

    cfg = HimoleConfig(aux_loss_weight=0.1, diversity_loss_weight=0.2)
    assert torch.allclose(combine_stage2_losses(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0), cfg), torch.tensor(1.8))
    assert maxvio_global([2, 2, 2]) == 0.0
    assert maxvio_global([4, 2, 0]) == 1.0


def test_deterministic_kmeans_and_subset_builder():
    cfg = HimoleConfig(num_kcgs=2, experts_per_kcg=1, top_k=1)
    embeddings = torch.tensor([[0.0], [0.1], [10.0], [10.1]])
    ids = cluster_training_examples(embeddings, cfg)
    subsets = build_cluster_subsets(["a", "b", "c", "d"], ids, cfg)

    assert ids.shape == (4,)
    assert sorted(len(subset) for subset in subsets) == [2, 2]


def test_stage_trainers_run_one_step_on_a_tiny_causal_model():
    cfg = HimoleConfig(
        num_kcgs=1,
        experts_per_kcg=2,
        top_k=1,
        lora_r=2,
        max_steps=1,
        stage1_max_steps=1,
        batch_size=2,
        micro_batch_size=1,
        diversity_every_n_layers=1,
    )
    model = _TinyCausalModel(cfg)
    tokenizer = _Tokenizer()

    state = train_single_kcg(model, _DATASET, cfg, group_id=0, tokenizer=tokenizer)
    assert set(state["projection"]) == {"lora_a.weight", "lora_b.weight"}
    assert all(tensor.device.type == "cpu" for tensor in state["projection"].values())

    result = train_himole_stage2(model, tokenizer, _DATASET, cfg)
    assert result["steps"] == 1
    assert model.gradient_checkpointing_enabled
    assert model.input_grads_enabled
    assert not model.config.use_cache
