"""
SOAPSpectralL1Reg: SOAP optimizer with spectral L1 (nuclear-norm) regularization.

After each SOAP update step, applies nuclear-norm proximal descent to 2D weight matrices,
analogous to AdamWSpectralL1Reg but using SOAP as the base optimizer.
"""

import torch

from .soap import SOAP


def zeropower_via_newtonschulz5(W, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of W.

    Returns an approximation of UV^T (where USV^T = W is the SVD), which serves as the
    subgradient of the nuclear norm at W. Used for the cheap spectral L1 regularization step.
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
        B = b * A + c * A @ A
        X = a * X + B @ X

    if W.size(0) > W.size(1):
        X = X.T
    return X


def _singular_value_threshold(W, tau):
    """Exact proximal operator of the nuclear norm: singular-value soft-thresholding.

        prox_{tau ||.||_*}(W) = U * diag(max(sigma_i - tau, 0)) * V^T

    Unlike the Newton-Schulz subgradient step (which subtracts tau*U V^T and only
    shifts the whole spectrum), this sets singular values below ``tau`` to exactly
    zero, producing genuine rank reduction. Only the surviving components are kept.
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


class SOAPSpectralL1Reg(SOAP):
    """SOAP with spectral L1 (nuclear-norm) regularization.

    After each SOAP update, applies nuclear-norm proximal descent to all 2D weight
    matrices.

    On most steps, uses the cheap Newton-Schulz subgradient approximation:
        W <- W - tau * zeropower(W)
    Every ``svt_interval`` steps (if > 0), uses exact singular-value soft-thresholding (SVT):
        W <- prox_{tau||.||_*}(W)
    which zeros sub-threshold singular values and produces real rank reduction.

    Arguments:
        params: Parameters to optimize.
        lr: Learning rate.
        betas: Adam betas (b1, b2).
        shampoo_beta: Beta for the Shampoo preconditioner moving average. -1 uses betas[1].
        eps: Adam epsilon for numerical stability.
        weight_decay: Weight decay coefficient.
        precondition_frequency: How often to update the preconditioner.
        max_precond_dim: Maximum dimension of the preconditioner.
        merge_dims: Whether to merge dimensions of the preconditioner.
        precondition_1d: Whether to precondition 1D gradients.
        normalize_grads: Whether to normalize gradients per layer.
        data_format: Data format for convolutional layers.
        correct_bias: Whether to use bias correction in Adam.
        spectral_l1_reg_coef: Nuclear-norm penalty strength.
        svt_interval: How often to do exact SVT. 0 = always use NS subgradient.
        svt_thresh: SVT threshold. If None, uses lr * spectral_l1_reg_coef.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
        spectral_l1_reg_coef: float = 0.1,
        svt_interval: int = 0,
        svt_thresh=None,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            merge_dims=merge_dims,
            precondition_1d=precondition_1d,
            normalize_grads=normalize_grads,
            data_format=data_format,
            correct_bias=correct_bias,
        )
        self.spectral_l1_reg_coef = spectral_l1_reg_coef
        self.svt_interval = svt_interval
        self.svt_thresh = svt_thresh
        self._global_step = 0

    @torch.no_grad()
    def step(self):
        # Run the standard SOAP step first
        result = super().step()

        self._global_step += 1
        step = self._global_step

        coef = self.spectral_l1_reg_coef
        if coef <= 0:
            return result

        for group in self.param_groups:
            lr = group["lr"]
            tau = self.svt_thresh if self.svt_thresh is not None else lr * coef

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    continue

                do_svt = self.svt_interval > 0 and step % self.svt_interval == 0
                if do_svt:
                    p.data.copy_(_singular_value_threshold(p.data, tau))
                else:
                    ns_approx = zeropower_via_newtonschulz5(p.data, steps=5)
                    p.data.add_(ns_approx, alpha=-tau)

        return result
