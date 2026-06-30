"""Small shared helpers: reproducibility, device, logging."""
import logging
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Return 'cuda' if a GPU is visible, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_logger(name: str) -> logging.Logger:
    """A logger that prints step/eval info to stdout."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(name)
