"""Hyperparameters for the Build-1 baseline and HiMoLE training."""
from dataclasses import dataclass


@dataclass
class BaselineConfig:
    # --- models ---
    base_model: str = "Qwen/Qwen2.5-7B"
    tiny_model: str = "sshleifer/tiny-gpt2"
    use_tiny: bool = False

    # --- LoRA ---
    lora_r: int = 80
    lora_alpha: int = 160
    lora_dropout: float = 0.05
    target_modules: tuple = ("up_proj", "down_proj", "gate_proj")

    # --- data ---
    id_dataset: str = "rajpurkar/squad"
    ood_dataset: str = "lucadiliello/newsqa"
    cutoff_len: int = 1024
    max_train_samples: int | None = None
    max_eval_samples: int | None = 500
    eval_batch_size: int = 4

    # --- optimization ---
    lr: float = 3e-4
    batch_size: int = 16
    micro_batch_size: int = 1
    max_steps: int = 10000
    eval_every: int = 50
    early_stop_patience: int = 10
    seed: int = 42

    output_dir: str = "outputs/baseline_lora"


@dataclass
class HimoleConfig:
    """Configuration for the EQA/SQuAD HiMoLE reproduction."""

    # --- task profile ---
    base_model: str = "Qwen/Qwen2.5-7B"
    tiny_model: str = "sshleifer/tiny-gpt2"
    use_tiny: bool = False
    id_dataset: str = "rajpurkar/squad"
    ood_dataset: str = "lucadiliello/newsqa"
    clustering_encoder: str = "microsoft/deberta-v3-large"

    # --- expert topology ---
    num_kcgs: int = 3
    experts_per_kcg: int = 4
    target_modules: tuple = ("up_proj", "down_proj", "gate_proj")

    # --- LoRA experts ---
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    top_k: int = 2

    # --- data and loop control ---
    cutoff_len: int = 1024
    max_train_samples: int | None = None
    max_eval_samples: int | None = 500
    eval_batch_size: int = 4
    batch_size: int = 16
    micro_batch_size: int = 1
    max_steps: int = 10000
    eval_every: int = 50
    early_stop_patience: int = 10
    seed: int = 42

    # --- two-stage optimization ---
    stage1_lr: float = 3e-4
    stage2_lr: float = 3e-5
    # The paper does not report alpha/beta. These are conservative starting
    # values and should be tuned for a reproduction run.
    aux_loss_weight: float = 0.01
    diversity_loss_weight: float = 0.01
    diversity_eps: float = 1e-8
    diversity_every_n_layers: int = 10
    clustering_batch_size: int = 16
    clustering_max_iterations: int = 100
    stage1_max_steps: int | None = None

    output_dir: str = "outputs/himole"
    stage1_output_dir: str = "outputs/himole/stage1"
    stage2_output_dir: str = "outputs/himole/stage2"

    @property
    def total_experts(self) -> int:
        """Return the total number of KCEs across all KCGs."""
        return self.num_kcgs * self.experts_per_kcg

    def validate(self) -> None:
        """Fail early when topology settings cannot match the paper equations."""
        if self.num_kcgs <= 0:
            raise ValueError("num_kcgs must be positive")
        if self.experts_per_kcg <= 0:
            raise ValueError("experts_per_kcg must be positive")
        if self.top_k <= 0 or self.top_k > self.total_experts:
            raise ValueError("top_k must be between 1 and total_experts")
        if self.lora_r <= 0:
            raise ValueError("lora_r must be positive")
        if self.clustering_batch_size <= 0:
            raise ValueError("clustering_batch_size must be positive")
        if self.clustering_max_iterations <= 0:
            raise ValueError("clustering_max_iterations must be positive")
        if self.diversity_every_n_layers <= 0:
            raise ValueError("diversity_every_n_layers must be positive")
        if self.batch_size <= 0 or self.micro_batch_size <= 0:
            raise ValueError("batch_size and micro_batch_size must be positive")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
