"""Small, hand-computable tests for the HiMoLE Stage-2 losses."""

import pytest
import torch

from himole.config import HimoleConfig
from himole.train.losses import (
    auxiliary_load_balancing_loss,
    combine_stage2_losses,
    diversity_loss,
)


def test_auxiliary_loss_is_one_for_uniform_probability_and_load():
    probabilities = torch.full((1, 4, 2), 0.5)
    topk_indices = torch.tensor([[[0], [1], [0], [1]]])

    loss = auxiliary_load_balancing_loss(probabilities, topk_indices, num_experts=2)

    assert torch.equal(loss, torch.tensor(1.0))


def test_auxiliary_loss_matches_skewed_hand_computation():
    probabilities = torch.tensor([[[0.8, 0.2], [0.6, 0.4]]])
    topk_indices = torch.tensor([[[0], [0]]])

    # F=[1, 0], P=[0.7, 0.3], E=2.
    loss = auxiliary_load_balancing_loss(probabilities, topk_indices, num_experts=2)

    assert torch.allclose(loss, torch.tensor(1.4))


def test_auxiliary_loss_retains_probability_gradients():
    probabilities = torch.tensor([[[0.8, 0.2], [0.6, 0.4]]], requires_grad=True)
    topk_indices = torch.tensor([[[0], [1]]])

    auxiliary_load_balancing_loss(probabilities, topk_indices, 2).backward()

    assert probabilities.grad is not None
    assert torch.count_nonzero(probabilities.grad) > 0


@pytest.mark.parametrize(
    ("probabilities", "indices", "message"),
    [
        (torch.ones(1, 1, 3), torch.zeros(1, 1, 1, dtype=torch.long), "expert dimension"),
        (torch.ones(1, 2, 2), torch.zeros(1, 1, 1, dtype=torch.long), "align"),
    ],
)
def test_auxiliary_loss_validates_router_shapes(probabilities, indices, message):
    with pytest.raises(ValueError, match=message):
        auxiliary_load_balancing_loss(probabilities, indices, num_experts=2)


def test_diversity_loss_averages_all_group_pairs():
    first = torch.tensor([[[1.0, 0.0]]])
    second = torch.tensor([[[0.0, 1.0]]])
    third = torch.tensor([[[-1.0, 0.0]]])

    # Pair similarities are 0, -1, and 0.
    loss = diversity_loss([first, second, third])

    assert torch.allclose(loss, torch.tensor(-1.0 / 3.0))


def test_diversity_loss_is_scalar_and_differentiable():
    first = torch.randn(2, 3, 4, requires_grad=True)
    second = torch.randn(2, 3, 4, requires_grad=True)

    loss = diversity_loss([first, second])
    loss.backward()

    assert loss.shape == ()
    assert first.grad is not None
    assert second.grad is not None


def test_diversity_loss_is_zero_when_fewer_than_two_groups_exist():
    output = torch.randn(1, 2, 3)

    assert torch.equal(diversity_loss([]), torch.tensor(0.0))
    assert torch.equal(diversity_loss([output]), output.new_zeros(()))


def test_combined_loss_uses_configured_weights_and_backpropagates():
    task = torch.tensor(2.0, requires_grad=True)
    auxiliary = torch.tensor(3.0, requires_grad=True)
    diverse = torch.tensor(5.0, requires_grad=True)
    cfg = HimoleConfig(aux_loss_weight=0.1, diversity_loss_weight=0.2)

    loss = combine_stage2_losses(task, auxiliary, diverse, cfg)
    loss.backward()

    assert torch.allclose(loss, torch.tensor(3.3))
    assert task.grad == 1.0
    assert auxiliary.grad == pytest.approx(0.1)
    assert diverse.grad == pytest.approx(0.2)
