import math
from collections.abc import Iterable
from typing import Any

import torch

from optim.adamw_spectral_L1_reg import (
    _singular_value_threshold,
    zeropower_via_newtonschulz5,
)


class GaLoreProjector:
    """One-sided gradient low-rank projector used by GaLore."""

    def __init__(
        self,
        rank: int,
        update_proj_gap: int = 200,
        scale: float = 1.0,
    ) -> None:
        if rank <= 0:
            raise ValueError("rank must be > 0")
        if update_proj_gap <= 0:
            raise ValueError("update_proj_gap must be > 0")
        if scale <= 0.0:
            raise ValueError("scale must be > 0")
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix: torch.Tensor | None = None
        self.project_right: bool | None = None

    @torch.no_grad()
    def project(self, gradient: torch.Tensor, step: int) -> torch.Tensor:
        if gradient.ndim != 2:
            raise ValueError("GaLore projection requires a 2D gradient")
        project_right = gradient.shape[0] >= gradient.shape[1]
        if (
            self.ortho_matrix is None
            or self.project_right != project_right
            or step % self.update_proj_gap == 0
        ):
            self.ortho_matrix = self._orthogonal_matrix(gradient, project_right)
            self.project_right = project_right

        if project_right:
            return gradient @ self.ortho_matrix.T
        return self.ortho_matrix.T @ gradient

    @torch.no_grad()
    def project_back(self, low_rank_update: torch.Tensor) -> torch.Tensor:
        if self.ortho_matrix is None or self.project_right is None:
            raise RuntimeError("project must be called before project_back")
        if self.project_right:
            update = low_rank_update @ self.ortho_matrix
        else:
            update = self.ortho_matrix @ low_rank_update
        return update.mul(self.scale)

    @torch.no_grad()
    def _orthogonal_matrix(
        self, matrix: torch.Tensor, project_right: bool
    ) -> torch.Tensor:
        rank = min(self.rank, min(matrix.shape))
        original_dtype = matrix.dtype
        u, _, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
        basis = vh[:rank, :] if project_right else u[:, :rank]
        return basis.to(dtype=original_dtype)


def build_galore_param_groups(
    param_groups: Iterable[dict[str, Any]],
    *,
    rank: int,
    update_proj_gap: int,
    scale: float,
) -> list[dict[str, Any]]:
    """Mark decay-enabled 2D weights for GaLore while preserving group options."""

    result: list[dict[str, Any]] = []
    for source_group in param_groups:
        group = dict(source_group)
        params = list(group.pop("params"))
        projection_enabled = source_group.get("weight_decay") != 0.0
        galore_params = [
            parameter
            for parameter in params
            if projection_enabled and parameter.ndim == 2
        ]
        galore_param_ids = {id(parameter) for parameter in galore_params}
        regular_params = [
            parameter for parameter in params if id(parameter) not in galore_param_ids
        ]

        if galore_params:
            result.append(
                {
                    **group,
                    "params": galore_params,
                    "galore": True,
                    "rank": rank,
                    "update_proj_gap": update_proj_gap,
                    "scale": scale,
                }
            )
        if regular_params:
            result.append({**group, "params": regular_params, "galore": False})
    return result


class GaLoreAdamW(torch.optim.Optimizer):
    """AdamW with low-rank GaLore moments and L2 or spectral weight decay.

    Parameters marked with ``galore=True`` keep Adam moments in the projected
    gradient space. Other parameters use ordinary full-rank AdamW moments.
    ``weight_decay_type='spectral'`` replaces decoupled L2 decay on GaLore
    matrices with the project's spectral L1 (nuclear-norm) regularizer.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        weight_decay_type: str = "l2",
        spectral_l1_reg_coef: float = 0.0,
        spectral_l1_reg_coupled: bool = False,
        svt_interval: int = 0,
        svt_thresh: float | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if spectral_l1_reg_coef < 0.0:
            raise ValueError("spectral_l1_reg_coef must be >= 0")
        if weight_decay_type not in {"l2", "spectral"}:
            raise ValueError("weight_decay_type must be 'l2' or 'spectral'")
        if spectral_l1_reg_coupled and weight_decay_type != "spectral":
            raise ValueError("coupled spectral regularization requires spectral decay")
        if svt_interval < 0:
            raise ValueError("svt_interval must be >= 0")
        if spectral_l1_reg_coupled and svt_interval != 0:
            raise ValueError("coupled spectral regularization does not support SVT")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decay_type=weight_decay_type,
            spectral_l1_reg_coef=spectral_l1_reg_coef,
            spectral_l1_reg_coupled=spectral_l1_reg_coupled,
            svt_interval=svt_interval,
            svt_thresh=svt_thresh,
            galore=False,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            for parameter in group["params"]:
                task_gradient = parameter.grad
                if task_gradient is None:
                    continue
                if task_gradient.is_sparse:
                    raise RuntimeError("GaLoreAdamW does not support sparse gradients")

                state = self.state[parameter]
                if "step" not in state:
                    state["step"] = 0
                step = state["step"]
                use_galore = group["galore"] and parameter.ndim == 2

                adam_gradient = task_gradient
                if (
                    use_galore
                    and group["weight_decay_type"] == "spectral"
                    and group["spectral_l1_reg_coupled"]
                    and group["spectral_l1_reg_coef"] > 0.0
                ):
                    spectral_gradient = zeropower_via_newtonschulz5(parameter, 5)
                    adam_gradient = task_gradient.add(
                        spectral_gradient, alpha=group["spectral_l1_reg_coef"]
                    )

                if use_galore:
                    if "projector" not in state:
                        state["projector"] = GaLoreProjector(
                            rank=group["rank"],
                            update_proj_gap=group["update_proj_gap"],
                            scale=group["scale"],
                        )
                    adam_gradient = state["projector"].project(adam_gradient, step)

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(adam_gradient)
                    state["exp_avg_sq"] = torch.zeros_like(adam_gradient)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] = step + 1
                exp_avg.lerp_(adam_gradient, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    adam_gradient, adam_gradient, value=1.0 - beta2
                )

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denominator = (
                    exp_avg_sq.sqrt()
                    .div_(math.sqrt(bias_correction2))
                    .add_(group["eps"])
                )
                update = exp_avg.div(denominator).div_(bias_correction1)
                if use_galore:
                    update = state["projector"].project_back(update)

                if group["weight_decay_type"] == "l2" and group["weight_decay"]:
                    parameter.mul_(1.0 - lr * group["weight_decay"])
                parameter.add_(update, alpha=-lr)

                if (
                    use_galore
                    and group["weight_decay_type"] == "spectral"
                    and not group["spectral_l1_reg_coupled"]
                    and group["spectral_l1_reg_coef"] > 0.0
                ):
                    do_svt = (
                        group["svt_interval"] > 0
                        and state["step"] % group["svt_interval"] == 0
                    )
                    if do_svt:
                        threshold = (
                            group["svt_thresh"]
                            if group["svt_thresh"] is not None
                            else lr * group["spectral_l1_reg_coef"]
                        )
                        parameter.copy_(_singular_value_threshold(parameter, threshold))
                    else:
                        spectral_gradient = zeropower_via_newtonschulz5(parameter, 5)
                        parameter.add_(
                            spectral_gradient,
                            alpha=-(lr * group["spectral_l1_reg_coef"]),
                        )

        return loss
