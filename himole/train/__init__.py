"""Training loops for the baseline and the two-stage HiMoLE scaffold."""

from himole.train.stage1 import initialize_kcgs, train_single_kcg
from himole.train.stage2 import train_himole_stage2

__all__ = ["initialize_kcgs", "train_himole_stage2", "train_single_kcg"]
