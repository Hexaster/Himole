"""Stage-2 HiMoLE auxiliary and diversity loss scaffold."""
import torch


def auxiliary_load_balancing_loss(router_probabilities, topk_indices, num_experts: int):
    """Compute the Switch/ST-MoE load-balancing loss L_aux."""
    if router_probabilities.shape[-1] != num_experts:
        raise ValueError("router_probabilities has the wrong expert dimension")
    if topk_indices.shape[:-1] != router_probabilities.shape[:-1]:
        raise ValueError("topk_indices must align with router_probabilities")
    dispatches = torch.nn.functional.one_hot(topk_indices, num_classes=num_experts)
    fraction_dispatched = dispatches.to(router_probabilities.dtype).sum(dim=tuple(range(dispatches.ndim - 1)))
    fraction_dispatched = fraction_dispatched / topk_indices.numel()
    mean_probability = router_probabilities.reshape(-1, num_experts).mean(dim=0)
    return num_experts * torch.sum(fraction_dispatched * mean_probability)


def diversity_loss(kcg_outputs, eps: float = 1e-8):
    """Compute L_diverse from pairwise normalized KCG outputs."""
    if len(kcg_outputs) < 2:
        if not kcg_outputs:
            return torch.tensor(0.0)
        return kcg_outputs[0].new_zeros(())
    flattened = [output.reshape(-1) for output in kcg_outputs]
    normalized = [output / output.norm(p=2).clamp_min(eps) for output in flattened]
    similarities = [
        torch.dot(normalized[left], normalized[right])
        for left in range(len(normalized))
        for right in range(left + 1, len(normalized))
    ]
    return torch.stack(similarities).mean()


def combine_stage2_losses(task_loss, aux_loss, diverse_loss, cfg):
    """Return L_task + alpha*L_aux + beta*L_diverse."""
    return task_loss + cfg.aux_loss_weight * aux_loss + cfg.diversity_loss_weight * diverse_loss
