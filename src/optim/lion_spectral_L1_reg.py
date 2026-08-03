"""Lion with spectral L1 (nuclear-norm) regularization."""

import torch

from .adamw_spectral_L1_reg import (
    _singular_value_threshold,
    zeropower_via_newtonschulz5,
)


class LionSpectralL1Reg(torch.optim.Optimizer):
    """Lion followed by the same spectral L1 proximal step used by AdamW.

    The Lion update is applied first. Matrix and convolution weights then receive
    either the Newton-Schulz nuclear-norm subgradient step or periodic exact SVT.
    Non-matrix parameters retain Lion's decoupled weight decay.
    """

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        spectral_l1_reg_coef=0.1,
        svt_interval=0,
        svt_thresh=None,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if svt_interval < 0:
            raise ValueError("svt_interval must be >= 0")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            spectral_l1_reg_coef=spectral_l1_reg_coef,
            svt_interval=svt_interval,
            svt_thresh=svt_thresh,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            spectral_l1_reg_coef = group["spectral_l1_reg_coef"]
            svt_interval = group["svt_interval"]
            svt_thresh = group["svt_thresh"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported!")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                exp_avg = state["exp_avg"]

                # Lion gradient step.
                orig_shape = p.shape
                if len(orig_shape) not in (2, 4):
                    p.mul_(1 - lr * weight_decay)
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(update.sign_(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                # Match AdamWSpectralL1Reg: spectral regularization replaces
                # ordinary weight decay for matrix and convolution weights.
                is_conv = len(orig_shape) == 4
                if (len(orig_shape) == 2 or is_conv) and spectral_l1_reg_coef > 0:
                    weight = p.view(orig_shape[0], -1) if is_conv else p
                    do_svt = (
                        svt_interval > 0 and state["step"] % svt_interval == 0
                    )
                    if do_svt:
                        tau = (
                            svt_thresh
                            if svt_thresh is not None
                            else lr * spectral_l1_reg_coef
                        )
                        new_weight = _singular_value_threshold(weight, tau)
                        p.copy_(new_weight.view(orig_shape) if is_conv else new_weight)
                    else:
                        nuclear_subgradient = zeropower_via_newtonschulz5(weight, 5)
                        p.add_(
                            nuclear_subgradient.view(orig_shape)
                            if is_conv
                            else nuclear_subgradient,
                            alpha=-(lr * spectral_l1_reg_coef),
                        )
        return loss
