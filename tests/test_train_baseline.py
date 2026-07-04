import pytest

from himole.config import BaselineConfig
from scripts.train_baseline import require_full_run_cuda


def test_real_run_requires_cuda():
    cfg = BaselineConfig()

    with pytest.raises(RuntimeError, match="CUDA is required"):
        require_full_run_cuda(cfg, "cpu")


def test_tiny_run_allows_cpu():
    cfg = BaselineConfig(use_tiny=True)

    require_full_run_cuda(cfg, "cpu")
