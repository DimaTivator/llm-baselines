import torch
from torch.optim import Optimizer


class SignSGD(Optimizer):
    """SGD where the update is the sign of the (optionally momentum-smoothed) gradient.

    update = -lr * sign(grad)                        # momentum=0
    update = -lr * sign(buf); buf = m*buf + grad     # momentum > 0

    Weight decay is applied in a decoupled (AdamW-style) manner:
        p = p * (1 - lr * wd) - lr * sign(...)
    This ensures the decay force is proportional to the weight magnitude and
    is not crushed by the sign operation (which is what happens with L2 / adding
    wd*p to the gradient before taking sign).
    """

    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if momentum != 0.0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = grad.clone()
                    else:
                        state["momentum_buffer"].mul_(momentum).add_(grad)
                    grad = state["momentum_buffer"]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(grad.sign(), alpha=-lr)

        return loss
