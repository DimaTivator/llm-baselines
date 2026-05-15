import math
import torch


def zeropower_via_newtonschulz5(W, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of W. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = W is the SVD.
    """
    # assert len(W.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = W
    if W.size(0) > W.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )
        X = a * X + B @ X

    if W.size(0) > W.size(1):
        X = X.T
    return X


class AdamWSpectralL1Reg(torch.optim.Optimizer):
    """Standard AdamW implementation copied from pytorch with added spectral L1 regularization using Newton-Schulz."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        spectral_l1_reg_coef=0.1,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            spectral_l1_reg_coef=spectral_l1_reg_coef,
        )

        super(AdamWSpectralL1Reg, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamWSpectralL1Reg, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            spectral_l1_reg_coef = group["spectral_l1_reg_coef"]

            for p in group["params"]:
                grad = p.grad

                if grad is None:
                    continue

                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported!")

                # Apply regularization to weight
                if len(p.data.shape) == 2:
                    l1_weight_reg = zeropower_via_newtonschulz5(p.data, 5)
                    p.data.mul_(1 - lr * wd)  # weight decay
                    p.data.add_(
                        l1_weight_reg, alpha=-(lr * spectral_l1_reg_coef)
                    )
                else:
                    p.data.mul_(1 - lr * wd)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["first_momentum"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["second_momentum"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                m = state["first_momentum"]
                v = state["second_momentum"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                m.lerp_(grad, 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                p.data.addcdiv_(m, denom, value=-(lr / bias_correction1))

        return loss
