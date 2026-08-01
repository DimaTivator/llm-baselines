"""SVD-LLM: Whitening-based SVD compression.

Paper / repo: https://github.com/AIoT-MLSys-Lab/SVD-LLM

SVD-LLM multiplies each weight matrix by the square-root of the input
activation covariance before performing SVD, aligning the decomposition with
the actual input distribution.  This yields a better low-rank approximation
in terms of reconstruction error weighted by the input statistics.

Typical usage:
    cov = collect_input_covariance(model, dataloader, n_batches=16)
    model = apply_svd_llm(model, rank=64, whitening_stats=cov)
"""

import torch


def collect_input_covariance(
    model: torch.nn.Module,
    dataloader,
    n_batches: int = 16,
    device: str = "cuda",
) -> dict:
    """Collect input covariance matrices for each linear layer.

    Args:
        model: the model to profile.
        dataloader: iterable of integer token batches, shape (B, T).
        n_batches: how many batches to average over.
        device: device to run forward passes on.

    Returns:
        dict mapping "<module_name>.weight" -> 2-D tensor of shape
        (in_features, in_features).
    """
    stats: dict = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        def _hook(mod, inp, out, _name=name):
            x = inp[0].detach().float()
            x_flat = x.reshape(-1, x.shape[-1])  # (N, in_features)
            cov = x_flat.T @ x_flat / x_flat.shape[0]
            key = _name + ".weight"
            if key not in stats:
                stats[key] = cov
            else:
                stats[key] += cov

        hooks.append(module.register_forward_hook(_hook))

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            model(batch)

    for h in hooks:
        h.remove()

    for k in stats:
        stats[k] = stats[k] / n_batches

    return stats


def apply_svd_llm(
    model: torch.nn.Module,
    rank,
    whitening_stats: dict,
    target_modules=("q_proj", "v_proj"),
    device: str = "cpu",
    margin: int = 0,
) -> tuple[torch.nn.Module, dict]:
    """SVD-LLM: whitening-aware SVD compression, applied in-place.

    Each target ``nn.Linear`` is replaced by a :class:`LowRankLinear` so the
    model genuinely shrinks (W ≈ A @ B with A: (out, r), B: (r, in)).

    Args:
        model: the model to compress (modified in-place).
        rank: number of singular values to retain, or the string ``"auto"``
            to use each layer's (rounded) effective rank.
        whitening_stats: dict from collect_input_covariance mapping
            "<module_name>.weight" to (in_features, in_features) covariance.
        target_modules: tuple of module-name suffixes to compress.
        device: device string.
        margin: only used with ``rank="auto"``; retained rank becomes
            ``round(effective_rank(W)) + margin`` (may be negative to compress
            more aggressively than the bare effective rank).

    Returns:
        ``(model, comp_info)`` where ``comp_info`` maps each compressed layer
        name to the rank it was reduced to.
    """
    from models.compress import LowRankLinear, effective_rank

    auto = isinstance(rank, str) and rank == "auto"
    comp_info: dict[str, int] = {}

    for parent_name, parent in list(model.named_modules()):
        for child_name, module in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not any(full_name.endswith(t) for t in target_modules):
                continue
            if not isinstance(module, torch.nn.Linear):
                continue

            W = module.weight.data.float()  # (out, in)
            out_f, in_f = W.shape
            max_rank = min(out_f, in_f)

            r = round(effective_rank(W)) + margin if auto else int(rank)
            r = max(1, min(r, max_rank))
            if r >= max_rank:
                continue  # no compression possible

            key = full_name + ".weight"
            if key in whitening_stats:
                X_cov = whitening_stats[key].float().to(W.device)  # (in, in)
                # Eigen-decompose covariance for stable sqrt / inverse-sqrt
                L, V = torch.linalg.eigh(
                    X_cov + 1e-6 * torch.eye(X_cov.shape[0], device=X_cov.device)
                )
                L = L.clamp(min=0.0)
                sqrt_cov = V @ torch.diag(L.sqrt()) @ V.T
                W_white = W @ sqrt_cov
            else:
                W_white = W
                sqrt_cov = None
                L = None
                V = None

            U, S, Vh = torch.linalg.svd(W_white, full_matrices=False)
            A_weight = U[:, :r] * S[:r].unsqueeze(0)  # (out, r)
            B_weight = Vh[:r, :]                       # (r, in)

            if sqrt_cov is not None:
                # Undo whitening on the right factor: W ≈ A @ (B @ inv_sqrt_cov)
                inv_sqrt_cov = V @ torch.diag(1.0 / (L.sqrt() + 1e-8)) @ V.T
                B_weight = B_weight @ inv_sqrt_cov

            bias = module.bias.data if module.bias is not None else None
            low_rank = LowRankLinear(in_f, out_f, r, bias=bias).to(W.device)
            with torch.no_grad():
                low_rank.A.weight.copy_(A_weight.to(module.weight.dtype))
                low_rank.B.weight.copy_(B_weight.to(module.weight.dtype))
            setattr(parent, child_name, low_rank)
            comp_info[full_name] = r

    tag = "auto" if auto else rank
    print(f"[SVD-LLM] Compressed {len(comp_info)} modules with rank={tag}.")
    return model, comp_info
