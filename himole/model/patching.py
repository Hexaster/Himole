"""Utilities for replacing base-model FFN modules with HiMoLE layers."""
from torch import nn

from himole.model.layer import HiMoLEFFNLayer


def freeze_base_model(model):
    """Freeze all parameters before adding trainable HiMoLE experts and routers."""
    # This chunk enforces the paper's parameter-efficient setup: W0 stays fixed,
    # while the KCE LoRA matrices and routers are trainable.
    for param in model.parameters():
        param.requires_grad = False
    return model


def find_target_ffn_modules(model, target_modules):
    """Yield (module_name, module) pairs whose names match configured FFN targets."""
    for name, module in model.named_modules():
        # GPT-2's c_proj name is also used by attention. Limit c_fc/c_proj matches
        # to its MLP container; Llama/Qwen projection names are already unique.
        is_gpt_mlp = (name.endswith("c_fc") or name.endswith("c_proj")) and (
            ".mlp." in name or name.startswith("mlp.")
        )
        if any(name.endswith(target) for target in target_modules) and (is_gpt_mlp or not name.endswith(("c_fc", "c_proj"))):
            yield name, module


def _module_dimensions(module):
    """Return input/output widths for Linear and HuggingFace Conv1D projections."""
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(module, "nf") and weight.ndim == 2:
        return weight.shape[0], weight.shape[1]
    raise TypeError(f"unsupported FFN projection type: {type(module).__name__}")


def _replace_child_module(model, module_name, replacement):
    """Install replacement at a dotted module path without relying on model internals."""
    parent = model
    *parents, child_name = module_name.split(".")
    for name in parents:
        parent = getattr(parent, name)
    setattr(parent, child_name, replacement)


def attach_himole_to_model(model, cfg):
    """Replace target FFN projections/modules with HiMoLEFFNLayer wrappers."""
    cfg.validate()
    freeze_base_model(model)
    targets = ("c_fc", "c_proj") if cfg.use_tiny else cfg.target_modules
    replacements = list(find_target_ffn_modules(model, targets))
    if not replacements:
        raise ValueError(f"no FFN projections found for targets {tuple(targets)!r}")

    for name, module in replacements:
        in_features, out_features = _module_dimensions(module)
        layer = HiMoLEFFNLayer(module, in_features, out_features, cfg)
        weight = getattr(module, "weight", None)
        if weight is not None:
            layer.to(device=weight.device, dtype=weight.dtype)
        _replace_child_module(model, name, layer)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"HiMoLE trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model
