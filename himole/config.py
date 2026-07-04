"""All Build-1 hyperparameters in one place. Values from the HiMoLE paper (Table 7)."""
from dataclasses import dataclass


@dataclass
class BaselineConfig:
    # --- models ---
    base_model: str = "Qwen/Qwen2.5-7B"  # real run (ungated, Apache-2.0; Llama-2-7B stand-in)
    tiny_model: str = "sshleifer/tiny-gpt2"        # smoke test / notebook
    use_tiny: bool = False                          # flip True for fast local runs

    # --- LoRA (plain-LoRA baseline matches HiMoLE param count) ---
    lora_r: int = 80
    lora_alpha: int = 160
    lora_dropout: float = 0.05
    # FFN projection names shared by Llama/Qwen-style MLPs; overridden for tiny-gpt2.
    target_modules: tuple = ("up_proj", "down_proj", "gate_proj")

    # --- data ---
    id_dataset: str = "rajpurkar/squad"
    ood_dataset: str = "microsoft/newsqa"
    cutoff_len: int = 1024
    max_train_samples: int | None = None  # None = full; set small for smoke test
    max_eval_samples: int | None = 500   # cap periodic/base-model evals; None = full val set

    # --- optim / loop ---
    lr: float = 3e-4          # paper Table 7 plain-LoRA learning rate
    batch_size: int = 16      # EFFECTIVE/global batch (paper Table 7); reached via accumulation
    micro_batch_size: int = 1  # per-step forward size that must fit in VRAM; accum = batch/micro
    max_steps: int = 10000    # counts OPTIMIZER steps, not micro-batches
    eval_every: int = 50
    early_stop_patience: int = 10  # number of stale evals before stopping
    seed: int = 42

    # --- io ---
    output_dir: str = "outputs/baseline_lora"
