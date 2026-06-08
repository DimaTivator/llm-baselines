# Lion optimizer — standalone implementation
# Based on: https://github.com/lucidrains/lion-pytorch
# Paper: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)
# https://arxiv.org/abs/2302.06675

from typing import Callable, Iterable, Tuple

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    r"""Implements the Lion (EvoLved Sign Momentum) optimizer.

    Lion uses the sign of the gradient-based update, making every parameter
    receive an update of the same magnitude. This leads to more uniform
    effective learning rates across all parameters.

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-4)
        betas (Tuple[float, float]): coefficients used for computing the
            update direction and the exponential moving average of the
            gradient. beta1 controls the update direction, beta2 controls
            the EMA that is stored in state (default: (0.9, 0.99))
        weight_decay (float): decoupled weight decay coefficient
            (default: 0.0)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the
                model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Lion does not support sparse gradients"
                    )

                state = self.state[p]

                # Lazy state initialisation
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Compute update direction: sign(beta1 * m + (1 - beta1) * g)
                update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1)
                p.add_(update.sign_(), alpha=-lr)

                # Update EMA: m <- beta2 * m + (1 - beta2) * g
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss
