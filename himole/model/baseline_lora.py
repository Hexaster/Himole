"""Load the base LLM (frozen) and attach trainable LoRA adapters with peft."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def load_tokenizer(cfg):
    """Load tokenizer; causal LMs often lack a pad token, so reuse eos."""
    name = cfg.tiny_model if cfg.use_tiny else cfg.base_model
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(cfg):
    """Load the frozen base model (bf16 on GPU, fp32 on CPU for the tiny model)."""
    name = cfg.tiny_model if cfg.use_tiny else cfg.base_model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return model


def attach_lora(model, cfg):
    """Wrap the base model with LoRA adapters on the FFN projections."""
    # tiny-gpt2 has no up/down/gate_proj; its MLP uses 'c_fc'/'c_proj'.
    targets = ["c_fc", "c_proj"] if cfg.use_tiny else list(cfg.target_modules)
    # TODO (the core learning point): build a LoraConfig with
    #   r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
    #   target_modules=targets, task_type="CAUSAL_LM", bias="none".
    #   Then peft_model = get_peft_model(model, lora_config).
    #   Finally peft_model.print_trainable_parameters() and return peft_model.
    raise NotImplementedError("TODO: build LoraConfig and wrap with get_peft_model")
