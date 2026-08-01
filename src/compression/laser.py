"""LASER: Low-rank Adaptation for Sparse Error Reduction.

Paper / repo: https://github.com/pratyushasharma/laser

LASER applies truncated SVD to selected weight matrices in a transformer and
replaces each weight with its rank-k approximation.  The compression is
applied in-place and requires no additional data.

Typical usage:
    model = apply_laser(model, rank=64, target_modules=("q_proj", "v_proj"))
"""

import torch


def apply_laser(
    model: torch.nn.Module,
    rank,
    target_modules=("q_proj", "v_proj"),
    device: str = "cpu",
) -> torch.nn.Module:
    """Apply LASER: truncated SVD compression to target modules in-place.

    Args:
        model: the model to compress (modified in-place).
        rank: number of singular values / vectors to retain, or the string
            ``"auto"`` to keep each layer's (rounded) effective rank.
        target_modules: tuple of module-name suffixes to compress.
        device: device string used to move intermediate tensors if needed.

    Returns:
        The same model object with compressed weights.
    """
    from models.compress import effective_rank

    auto = isinstance(rank, str) and rank == "auto"
    compressed = 0
    for name, module in model.named_modules():
        if not any(name.endswith(t) for t in target_modules):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue

        W = module.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = round(effective_rank(W)) if auto else int(rank)
        r = max(1, min(r, S.numel()))
        W_compressed = (U[:, :r] * S[:r]) @ Vh[:r, :]
        module.weight.data = W_compressed.to(module.weight.dtype)
        compressed += 1

    tag = "auto" if auto else rank
    print(f"[LASER] Compressed {compressed} modules with rank={tag}.")
    return model
