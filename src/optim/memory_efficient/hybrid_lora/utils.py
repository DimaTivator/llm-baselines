import gc
from typing import List, Optional

import torch
import torch.nn as nn

from .hybrid_lora_linear import HybridLoRALinear, HybridRiemannianLoRALinear


_SCOPE_KEYWORDS = {
    "all":  ["attn", "mlp"],
    "attn": ["attn"],
    "mlp":  ["mlp"],
}

# Parameter names containing any of these keywords skip weight decay.
_NO_DECAY_KW = ("norm", "bias", "embed", "lm_head")


def apply_hybrid_lora(
    model: nn.Module,
    rank: int,
    alpha: float = 1.0,
    scope: str = "all",
    target_modules: Optional[List[str]] = None,
    riemannian_lora: bool = False,
    verbose: bool = True,
) -> None:
    """
    Replace matching nn.Linear layers with HybridLoRALinear in-place.

    Must be called BEFORE DDP wrapping.

    Args:
        model:            Model to modify.
        rank:             LoRA rank r.
        alpha:            LoRA scaling factor (effective scale = alpha / rank).
                          Ignored when riemannian_lora=True (scale is implicit).
        scope:            "all" | "attn" | "mlp" — broad filter by module-name keywords.
        target_modules:   Optional list of substrings matched against the child
                          module name (e.g. ["q_proj", "k_proj"]).  When provided,
                          only layers whose name contains at least one entry are
                          replaced.  When None, every Linear in ``scope`` is replaced.
        riemannian_lora:  If True, replace with HybridRiemannianLoRALinear (Stiefel
                          manifold parametrization) instead of HybridLoRALinear.
        verbose:          Print a line per replacement.
    """
    keywords = _SCOPE_KEYWORDS[scope]
    full_params = sum(p.numel() for p in model.parameters())

    replacements = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not any(kw in full_name for kw in keywords):
                continue
            if target_modules is not None:
                if not any(tm in child_name for tm in target_modules):
                    continue
            if riemannian_lora:
                new_mod = HybridRiemannianLoRALinear(child, rank=rank)
            else:
                new_mod = HybridLoRALinear(child, rank=rank, alpha=alpha)
            replacements.append((parent, child_name, child, new_mod, full_name))

    for parent, child_name, old, new, full_name in replacements:
        setattr(parent, child_name, new)
        if verbose:
            print(f"HybridLoRA: {full_name}: {old} → {new}")
        del old
        torch.cuda.empty_cache()
        gc.collect()

    adapter_type = "Riemannian" if riemannian_lora else "standard"
    n_replaced = len(replacements)
    lr_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nHybridLoRA ({adapter_type}) applied: {n_replaced} layers, rank={rank}, scope={scope}"
        + (f", target_modules={target_modules}" if target_modules else "")
        + f"\n  Base params  : {full_params / 1e6:.2f} M"
        + f"\n  Total params : {lr_params / 1e6:.2f} M"
        + f"  (+{(lr_params - full_params) / 1e6:.2f} M LoRA adapters)\n"
    )


def get_hybrid_lora_param_groups(
    model: nn.Module,
    weight_decay: float,
    base_lr: float,
    lora_lr: float,
):
    """
    Build (base_groups, lora_groups) for HybridLoRAOptimizer.

    Handles both HybridLoRALinear and HybridRiemannianLoRALinear.

    For HybridRiemannianLoRALinear modules, lora groups include the
    ``param_names`` key required by the Riemannian optimizer.

    - base_groups : all model parameters except LoRA adapter factors,
                    split into decay / no-decay subgroups.
    - lora_groups : adapter factors of every Hybrid*LoRALinear module,
                    marked with ``is_lora_param=True`` and weight_decay=0.

    Args:
        model:        The (pre-DDP) model with Hybrid*LoRALinear modules.
        weight_decay: Applied to base weight-decay group; LoRA factors get 0.
        base_lr:      Learning rate for base param groups.
        lora_lr:      Learning rate for LoRA adapter param groups.

    Returns:
        (base_groups, lora_groups) — lists of param-group dicts.
    """
    lora_param_ids: set = set()
    lora_groups = []

    for _name, module in model.named_modules():
        if isinstance(module, HybridRiemannianLoRALinear):
            a_w = module.lora_A.weight
            b_w = module.lora_B.weight
            lora_param_ids.add(id(a_w))
            lora_param_ids.add(id(b_w))
            lora_groups.append({
                "params": [a_w, b_w],
                "param_names": ["lora_A.weight", "lora_B.weight"],
                "is_lora_param": True,
                "weight_decay": 0.0,
                "lr": lora_lr,
            })
        elif isinstance(module, HybridLoRALinear):
            lora_param_ids.add(id(module.lora_A))
            lora_param_ids.add(id(module.lora_B))
            lora_groups.append({
                "params": [module.lora_A, module.lora_B],
                "is_lora_param": True,
                "weight_decay": 0.0,
                "lr": lora_lr,
            })

    decay_params, no_decay_params = [], []
    for pname, p in model.named_parameters():
        if id(p) in lora_param_ids:
            continue
        if any(kw in pname for kw in _NO_DECAY_KW):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    base_groups = []
    if decay_params:
        base_groups.append({
            "params": decay_params,
            "weight_decay": weight_decay,
            "lr": base_lr,
        })
    if no_decay_params:
        base_groups.append({
            "params": no_decay_params,
            "weight_decay": 0.0,
            "lr": base_lr,
        })

    n_lora = sum(p.numel() for g in lora_groups for p in g["params"])
    n_base = sum(p.numel() for g in base_groups for p in g["params"])
    print(
        f"HybridLoRA param groups: "
        f"{n_base / 1e6:.2f}M base (lr={base_lr}), "
        f"{n_lora / 1e6:.2f}M LoRA (lr={lora_lr})"
    )
    return base_groups, lora_groups
