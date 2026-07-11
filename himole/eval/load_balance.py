"""Load-balance metrics for routed HiMoLE experts."""
import torch


def maxvio_global(expert_loads):
    """Compute the paper's MaxVio_global load-balance metric."""
    loads = torch.as_tensor(expert_loads, dtype=torch.float32)
    if loads.ndim != 1 or loads.numel() == 0:
        raise ValueError("expert_loads must be a non-empty one-dimensional sequence")
    expected_load = loads.mean()
    if expected_load == 0:
        return 0.0
    return ((loads.max() - expected_load) / expected_load).item()
