from typing import Callable, Type

import torch
from torch.optim import Optimizer


class HybridLoRAOptimizer(Optimizer):
    """
    Dual-optimizer wrapper for hybrid full-weight + LoRA adapter training.

    Routes parameters to two independent sub-optimizers:
      - base_optimizer  : intended for original model weights (e.g. stateless SGD / Lion)
      - lora_optimizer  : intended for LoRA adapter factors  (e.g. stateful AdamW)

    Both optimizers are constructed internally from their classes and kwargs,
    so this object owns the full optimizer lifecycle (step, state, checkpointing).

    Exposes a combined ``param_groups`` list so PyTorch LR schedulers work
    out of the box.  The group dicts are shared by reference, so scheduler
    LR updates propagate into both sub-optimizers automatically.

    Args:
        base_groups:     Param-group dicts for base model weights.
        lora_groups:     Param-group dicts for LoRA adapter parameters.
        base_opt_cls:    Optimizer class for base weights.
        lora_opt_cls:    Optimizer class for LoRA weights.
        base_opt_kwargs: kwargs forwarded verbatim to ``base_opt_cls``.
        lora_opt_kwargs: kwargs forwarded verbatim to ``lora_opt_cls``.
    """

    def __init__(
        self,
        base_groups,
        lora_groups,
        base_opt_cls: Type[Optimizer],
        lora_opt_cls: Type[Optimizer],
        base_opt_kwargs: dict,
        lora_opt_kwargs: dict,
    ):
        torch._C._log_api_usage_once(f"python.optimizer.{self.__class__.__name__}")
        self.defaults = {}
        self.state = {}

        self._base_opt = base_opt_cls(base_groups, **base_opt_kwargs)
        self._lora_opt = lora_opt_cls(lora_groups, **lora_opt_kwargs)

        # Combined view: group dicts are shared objects, not copies.
        self.param_groups = self._base_opt.param_groups + self._lora_opt.param_groups

        n_base = sum(p.numel() for g in self._base_opt.param_groups for p in g["params"])
        n_lora = sum(p.numel() for g in self._lora_opt.param_groups for p in g["params"])
        print(
            f"HybridLoRAOptimizer: "
            f"{n_base / 1e6:.2f}M base params → {base_opt_cls.__name__}, "
            f"{n_lora / 1e6:.2f}M LoRA params  → {lora_opt_cls.__name__}"
        )

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        self._base_opt.step()
        self._lora_opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self._base_opt.zero_grad(set_to_none=set_to_none)
        self._lora_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "base_optimizer": self._base_opt.state_dict(),
            "lora_optimizer": self._lora_opt.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self._base_opt.load_state_dict(state_dict["base_optimizer"])
        self._lora_opt.load_state_dict(state_dict["lora_optimizer"])
        # Re-sync the combined view after state restoration.
        self.param_groups = self._base_opt.param_groups + self._lora_opt.param_groups
