"""SliceGPT: PCA-based dimension slicing for transformer compression.

Paper / repo: https://github.com/microsoft/TransformerCompression

The full SliceGPT algorithm applies a consistent orthogonal rotation across
ALL layers of the transformer (including embeddings, attention projections,
MLP weights, and layer norms) so that the last (d - k) dimensions of every
hidden representation carry minimal variance.  Slicing those dimensions then
reduces the hidden size uniformly, yielding a smaller model architecture.

NOTE: The complete algorithm requires invasive model-architecture surgery
(shared rotation matrices inserted between every layer, modified embedding
projections, etc.) and is tightly coupled to a specific model layout.
Implementing it faithfully here would duplicate the TransformerCompression
library and break the project's "no external pip deps" constraint.

This module therefore provides a *simplified* SliceGPT-style approximation:
for each target linear layer, we run SVD and keep the top
  k = int((1 - slice_fraction) * min(m, n))
singular components — equivalent to LASER but motivated by the PCA/slicing
framing.  The function signature and docstring are designed to be a drop-in
placeholder that can be replaced with the full implementation when the model
architecture supports it.
"""

import torch


def apply_slice_gpt(
    model: torch.nn.Module,
    slice_fraction: float = 0.1,
    target_modules=("q_proj", "v_proj", "c_fc", "c_proj"),
    n_batches_pca: int = 16,
    dataloader=None,
    device: str = "cuda",
    rank=None,
) -> torch.nn.Module:
    """Simplified SliceGPT: PCA-based row slicing via per-layer truncated SVD.

    For each eligible linear layer, compute SVD of W and retain only the top
    ``int((1 - slice_fraction) * min(m, n))`` singular components.  This
    reduces the effective rank of each weight matrix without changing the
    model's tensor shapes (i.e. it is a weight-approximation rather than a
    true architectural slicing).

    For the full SliceGPT algorithm (which *does* shrink hidden dimensions
    across all layers consistently) see:
        https://github.com/microsoft/TransformerCompression

    Args:
        model: the model to compress (modified in-place).
        slice_fraction: fraction of singular components to remove (0.1 = 10%).
        target_modules: tuple of module-name suffixes to compress.
        n_batches_pca: unused in the simplified version; kept for API
            compatibility with the full implementation.
        dataloader: unused in the simplified version.
        device: device string.

    Returns:
        The same model with compressed weights.
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
        m, n = W.shape
        if rank is not None:
            # rank overrides slice_fraction: "auto" -> per-layer effective rank
            keep = round(effective_rank(W)) if auto else int(rank)
        else:
            keep = int((1.0 - slice_fraction) * min(m, n))
        keep = max(1, min(keep, min(m, n)))

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        W_approx = (U[:, :keep] * S[:keep]) @ Vh[:keep, :]
        module.weight.data = W_approx.to(module.weight.dtype)
        compressed += 1

    detail = ("rank=auto (per-layer effective rank)" if auto
              else f"rank={rank}" if rank is not None
              else f"slice_fraction={slice_fraction}, kept {100*(1-slice_fraction):.0f}% of singular values")
    print(
        f"[SliceGPT-simplified] Compressed {compressed} modules ({detail}). "
        "Note: full architectural slicing requires TransformerCompression library."
    )
    return model
