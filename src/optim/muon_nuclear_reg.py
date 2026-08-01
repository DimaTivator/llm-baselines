"""
MuonNuclearReg: Muon optimizer with nuclear-norm regularization injected into the
gradient before the momentum/orthogonalization step (in contrast to
``muon_spectral_L1_reg.py``, which applies the regularization as a separate
step *after* the Muon update).

For R(W) = coef * ||W||_*, the subgradient is U V^T (the outer product of the
left/right singular vectors of W). We approximate U V^T cheaply with the same
Newton-Schulz iteration Muon already uses for orthogonalization, applied to W
itself:

    M_t = beta * M_{t-1} + (1 - beta) * (grad_W L(W_t) + lambda * NS(W_t))
    W_{t+1} = W_t - eta * NS(M_t)

Non-2D params (embeddings, layernorm, lm_head, ...) fall back to plain AdamW,
unaffected by the regularization.
"""

import os

import torch
import torch.distributed as dist

from .muon import zeropower_via_newtonschulz5


def _singular_value_threshold(W, tau):
    """Exact proximal operator of the nuclear norm: singular-value soft-thresholding.

        prox_{tau ||.||_*}(W) = U * diag(max(sigma_i - tau, 0)) * V^T

    Unlike the Newton-Schulz subgradient used on most steps (which only shifts
    the whole spectrum down by approximately tau), this sets singular values
    below ``tau`` to exactly zero, producing genuine rank reduction.
    """
    orig_dtype = W.dtype
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    S = torch.clamp(S - tau, min=0.0)
    keep = S > 0
    if not torch.any(keep):
        return torch.zeros_like(W)
    U = U[:, keep]
    S = S[keep]
    Vh = Vh[keep, :]
    return ((U * S.unsqueeze(0)) @ Vh).to(orig_dtype)


class MuonNuclearReg(torch.optim.Optimizer):
    """Muon with nuclear-norm regularization folded into the momentum update.

    Arguments:
        muon_params: Parameters to be optimized by Muon (>=2D, non-embedding-like).
        lr: Muon learning rate (spectral norm of updates). (0.02 is a good default)
        momentum: EMA momentum coefficient, beta in the formula above.
        nesterov: Whether to use Nesterov-style look-ahead momentum.
        ns_steps: Newton-Schulz iterations used to orthogonalize the momentum M_t.
        adamw_params: Parameters for AdamW fallback (1D, embeddings, head).
        muon_wd: Decoupled L2 weight decay for Muon params (lambda in the DMuon formula).
        adamw_lr: Learning rate for the internal AdamW.
        adamw_betas: Betas for the internal AdamW.
        adamw_eps: Epsilon for the internal AdamW.
        adamw_wd: Weight decay for the internal AdamW.
        spectral_l1_reg_coef: Nuclear-norm penalty strength (lambda in the formula).
        nuclear_reg_ns_steps: Newton-Schulz iterations used to approximate U V^T
            of the weight matrix W itself (cheaper than the main ns_steps is fine).
        svt_interval: Every this many steps, skip the cheap Newton-Schulz subgradient
            (lambda * NS(W_t) folded into the gradient) and instead apply an exact
            proximal step (singular-value soft-thresholding) directly to the updated
            weights, analogous to ``adamw_spectral_L1_reg``'s ``svt_interval``. 0 disables
            it and always uses the Newton-Schulz subgradient.
        svt_thresh: Absolute singular-value threshold for the periodic SVT step. If
            None, uses lr * spectral_l1_reg_coef * svt_interval (so the average
            per-step shrinkage matches the skipped subgradient steps).
    """

    def __init__(
        self,
        muon_params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=6,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0,
        muon_wd=0.0,
        spectral_l1_reg_coef=0.1,
        nuclear_reg_ns_steps=5,
        svt_interval=0,
        svt_thresh=None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr / lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
            muon_wd=muon_wd,
            spectral_l1_reg_coef=spectral_l1_reg_coef,
            nuclear_reg_ns_steps=nuclear_reg_ns_steps,
            svt_interval=svt_interval,
            svt_thresh=svt_thresh,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and
            # doesn't look like an embedding or head layer
            self.state[p]["use_muon"] = p.ndim >= 2 and p.size(0) < 10000
        for p in adamw_params:
            self.state[p]["use_muon"] = False

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

        self._global_step = 0

    @torch.no_grad()
    def step(self):
        self._global_step += 1
        global_step = self._global_step

        for group in self.param_groups:
            ############################
            #    Muon + nuclear reg    #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            muon_wd = group["muon_wd"]
            coef = group["spectral_l1_reg_coef"]
            reg_ns_steps = group["nuclear_reg_ns_steps"]
            svt_interval = group["svt_interval"]
            svt_thresh = group["svt_thresh"]

            do_svt = coef > 0 and svt_interval > 0 and global_step % svt_interval == 0

            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(
                total_params, device="cuda", dtype=torch.bfloat16
            )
            curr_idx = 0
            for i, p in enumerate(params):
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    assert g is not None

                    # On most steps, fold the cheap Newton-Schulz subgradient
                    # approximation of the nuclear-norm penalty into the gradient.
                    # On SVT steps, skip it here -- the exact proximal step is
                    # applied directly to the weights after the Muon update below.
                    if coef > 0 and not do_svt:
                        w = p.data
                        if w.ndim > 2:
                            w = w.view(w.size(0), -1)
                        uv_approx = zeropower_via_newtonschulz5(w, steps=reg_ns_steps)
                        g = g + coef * uv_approx.to(g.dtype)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.lerp_(g, 1 - momentum)
                    g = buf.lerp(g, momentum) if group["nesterov"] else buf

                    g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    g *= 0.2 * max(g.size(0), g.size(1)) ** 0.5
                    updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
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
                if muon_wd != 0:
                    p.data.mul_(1 - lr * muon_wd)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

            # Periodic exact proximal step (singular-value soft-thresholding) on
            # the just-updated weights, replacing the per-step subgradient with
            # genuine rank reduction every svt_interval steps.
            if do_svt:
                tau = (
                    svt_thresh
                    if svt_thresh is not None
                    else lr * coef * svt_interval
                )
                for p in params:
                    if p.data.ndim == 2:
                        p.data.copy_(_singular_value_threshold(p.data, tau))

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr_adamw = group["adamw_lr_ratio"] * group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr_adamw * weight_decay)
                p.data.add_(g, alpha=-lr_adamw / scale)
