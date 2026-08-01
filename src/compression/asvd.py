"""ASVD: Activation-aware SVD for LLM compression.

Paper / repo: https://github.com/hahnyuan/ASVD4LLM

ASVD scales weight columns by input-activation norms before performing SVD,
then un-scales after reconstruction.  This makes the truncation error
proportional to how much each input dimension is actually used, giving
better preserved accuracy than plain SVD at the same rank.

Typical usage:
    stats = collect_activation_stats(model, dataloader, n_batches=16)
    model = apply_asvd(model, rank=64, activation_stats=stats)
"""

import torch


def collect_activation_stats(
    model: torch.nn.Module,
    dataloader,
    n_batches: int = 16,
    device: str = "cuda",
) -> dict:
    """Collect mean absolute input-activation norms for each linear layer.

    Args:
        model: the model to profile.
        dataloader: iterable of integer token batches, shape (B, T).
        n_batches: how many batches to average over.
        device: device to run forward passes on.

    Returns:
        dict mapping "<module_name>.weight" -> 1-D tensor of shape (in_features,).
    """
    stats: dict = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        def _hook(mod, inp, out, _name=name):
            x = inp[0].detach().float()
            # Average absolute value across all non-feature dimensions
            norm = x.abs().mean(dim=list(range(x.ndim - 1)))  # (in_features,)
            key = _name + ".weight"
            if key not in stats:
                stats[key] = norm
            else:
                stats[key] += norm

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


def apply_asvd(
    model: torch.nn.Module,
    rank,
    activation_stats: dict,
    alpha: float = 0.5,
    target_modules=("q_proj", "v_proj"),
    device: str = "cpu",
) -> tuple:
    """ASVD: activation-aware SVD compression applied in-place.

    Args:
        model: the model to compress.
        rank: number of singular values to retain, or the string ``"auto"``
            to use each layer's (rounded) effective rank.
        activation_stats: dict from collect_activation_stats mapping
            "<module_name>.weight" to a 1-D tensor of input activation norms.
        alpha: exponent for scaling (0 = no scaling, 1 = full scaling).
        target_modules: tuple of module-name suffixes to compress.
        device: device string (used for moving stats tensors).

    Returns:
        ``(model, comp_info)`` where ``comp_info`` maps each compressed layer
        name to the rank it was reduced to.
    """
    from models.compress import effective_rank as compute_effective_rank

    auto = isinstance(rank, str) and rank == "auto"
    comp_info: dict = {}

    for name, module in model.named_modules():
        if not any(name.endswith(t) for t in target_modules):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue

        W = module.weight.data.float()  # (out, in)
        out_f, in_f = W.shape
        max_rank = min(out_f, in_f)
        key = name + ".weight"

        if key in activation_stats:
            s = activation_stats[key].float().to(W.device)
            s = s ** alpha
            W_scaled = W * s.unsqueeze(0)  # scale columns
        else:
            W_scaled = W
            s = None

        U, S, Vh = torch.linalg.svd(W_scaled, full_matrices=False)

        if auto:
            r = max(1, min(round(compute_effective_rank(W_scaled)), max_rank))
        else:
            r = max(1, min(int(rank), max_rank))

        if r >= max_rank:
            continue  # no compression possible

        W_approx = (U[:, :r] * S[:r]) @ Vh[:r, :]

        if s is not None:
            # Undo column scaling
            W_approx = W_approx / s.unsqueeze(0)

        module.weight.data = W_approx.to(module.weight.dtype)
        comp_info[name] = r

    tag = "auto" if auto else rank
    print(f"[ASVD] Compressed {len(comp_info)} modules with rank={tag}, alpha={alpha}.")
    return model, comp_info
