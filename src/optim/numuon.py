"""
nuMuon: Nuclear-Norm-Constrained Muon for Compressible LLM Training
Paper: https://arxiv.org/abs/2603.03597

nuMuon modifies Muon by constraining the update direction to a nuclear-norm ball,
encouraging learned weights to develop low-rank structure. Instead of orthogonalizing
all singular directions (Muon), nuMuon uses only the top-k, where k is controlled by
a rank fraction schedule that can anneal during training.
"""

import math
import os

import torch
import torch.distributed as dist


def _numuon_svd_update(G, k, niter=2):
    """
    Compute top-k singular direction update U_k @ V_k^T via randomized SVD.
    """
    m, n = G.shape
    k = max(1, min(k, min(m, n)))
    U, S, V = torch.svd_lowrank(G, q=k, niter=niter)
    return U @ V.T


def _current_rank_fraction(r_start, r_end, schedule, step, total_steps):
    if r_end is None or schedule == "fixed" or total_steps is None or total_steps <= 0:
        return r_start
    t = min(step / total_steps, 1.0)
    if schedule == "cosine":
        return r_end + 0.5 * (r_start - r_end) * (1.0 + math.cos(math.pi * t))
    elif schedule == "linear":
        return r_start + (r_end - r_start) * t
    return r_start


class NuMuon(torch.optim.Optimizer):
    """
    nuMuon - Nuclear-Norm-Constrained Muon

    Paper: NuMuon: Nuclear-Norm-Constrained Muon for Compressible LLM Training
    https://arxiv.org/abs/2603.03597

    nuMuon solves a linear minimization oracle over the feasible set
    {ΔW | ||ΔW||_2 ≤ ρ, ||ΔW||_* ≤ τ}, whose closed-form solution is
    ΔW* = -ρ * U_k V_k^T where k = floor(τ/ρ). In practice ρ is absorbed
    into the learning rate and k is expressed as a fraction of min(d_in, d_out).

    Setting rank_fraction=1.0 (fixed) recovers approximately the same updates
    as Muon (all singular directions used). Lower rank fractions produce
    lower-rank weight updates and more compressible trained models.

    Arguments:
        muon_params: Parameters to be optimized by nuMuon (>=2D, non-embedding).
        lr: Learning rate. (0.02 is a good default, same as Muon)
        momentum: SGD momentum coefficient. (0.95 is a good default)
        nesterov: Use Nesterov-style momentum. (recommended)
        rank_fraction: Initial fraction of singular directions to use (0 < r <= 1).
        rank_fraction_final: Final rank fraction after schedule. None = fixed.
        rank_schedule: Annealing schedule for rank: 'fixed', 'cosine', or 'linear'.
        total_steps: Total training iterations (required for non-fixed schedule).
        svd_niter: Randomized SVD power iterations. 2 is sufficient for most cases.
        adamw_params: Parameters to be optimized by AdamW (1D, embeddings, lm_head).
        adamw_lr: Learning rate for the internal AdamW.
        adamw_betas: Betas for the internal AdamW.
        adamw_eps: Epsilon for the internal AdamW.
        adamw_wd: Weight decay for the internal AdamW.
    """

    def __init__(
        self,
        muon_params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        rank_fraction=1.0,
        rank_fraction_final=None,
        rank_schedule="cosine",
        total_steps=None,
        svd_niter=2,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            rank_fraction=rank_fraction,
            rank_fraction_final=rank_fraction_final,
            rank_schedule=rank_schedule,
            total_steps=total_steps,
            svd_niter=svd_niter,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr / lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        for p in muon_params:
            # Apply nuMuon to >=2D params that are not large embedding-like matrices
            self.state[p]["use_numuon"] = p.ndim >= 2 and p.size(0) < 10000
        for p in adamw_params:
            self.state[p]["use_numuon"] = False

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

    def step(self):
        for group in self.param_groups:

            # ---- nuMuon update for matrix params ----
            params = [p for p in group["params"] if self.state[p]["use_numuon"]]
            lr = group["lr"]
            momentum = group["momentum"]
            svd_niter = group["svd_niter"]

            if "step" not in group:
                group["step"] = 0
            current_step = group["step"]

            r = _current_rank_fraction(
                group["rank_fraction"],
                group["rank_fraction_final"],
                group["rank_schedule"],
                current_step,
                group["total_steps"],
            )

            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(
                total_params, device="cuda", dtype=torch.bfloat16
            )
            curr_idx = 0

            for i, p in enumerate(params):
                if i % self.world_size == self.rank:
                    g = p.grad
                    assert g is not None
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if group["nesterov"] else buf.clone()

                    m, n = g.shape
                    k = max(1, math.ceil(r * min(m, n)))

                    # Top-k SVD update: U_k @ V_k^T
                    update = _numuon_svd_update(g.bfloat16(), k, niter=svd_niter)

                    # Normalize to match Muon's update magnitude: scale by
                    # max(1, m/n)^0.5 * sqrt(min(m,n)/k) so that at rank_fraction=1
                    # the RMS matches Muon, and the magnitude stays consistent for
                    # different rank fractions.
                    scale = max(1.0, m / n) ** 0.5 * (min(m, n) / k) ** 0.5
                    update = update * scale

                    updates_flat[curr_idx : curr_idx + p.numel()] = update.flatten()
                curr_idx += p.numel()

            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            for p in params:
                g = (
                    updates_flat[curr_idx : curr_idx + p.numel()]
                    .view_as(p.data)
                    .type_as(p.data)
                )
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

            group["step"] += 1

            # ---- AdamW for non-nuMuon params (1D, embeddings, lm_head) ----
            params_adamw = [
                p for p in group["params"] if not self.state[p]["use_numuon"]
            ]
            lr_adamw = group["adamw_lr_ratio"] * group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params_adamw:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "adamw_step" not in state:
                    state["adamw_step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["adamw_step"] += 1
                step = state["adamw_step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                g_upd = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                bc_scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr_adamw * weight_decay)
                p.data.add_(g_upd, alpha=-lr_adamw / bc_scale)
