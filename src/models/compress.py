import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    """Approximates a Linear layer as two smaller linear layers via truncated SVD.

    W ≈ U_r @ diag(S_r) @ Vh_r  is stored as:
        A: (out_features, rank)  — weight = U_r @ diag(S_r)
        B: (rank, in_features)   — weight = Vh_r
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias=None):
        super().__init__()
        self.B = nn.Linear(in_features, rank, bias=False)
        self.A = nn.Linear(rank, out_features, bias=bias is not None)
        if bias is not None:
            with torch.no_grad():
                self.A.bias.copy_(bias)

    def forward(self, x):
        return self.A(self.B(x))


def _svd_compress_linear(layer: nn.Linear, rank: int) -> LowRankLinear:
    W = layer.weight.data.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # U: (out, k), S: (k,), Vh: (k, in)  where k = min(out, in)
    rank = min(rank, S.shape[0])

    A_weight = U[:, :rank] * S[:rank].unsqueeze(0)  # (out, rank)
    B_weight = Vh[:rank, :]                          # (rank, in)

    bias = layer.bias.data if layer.bias is not None else None
    device = layer.weight.device
    low_rank = LowRankLinear(
        layer.in_features, layer.out_features, rank, bias=bias
    ).to(device)
    with torch.no_grad():
        low_rank.A.weight.copy_(A_weight.to(device=device, dtype=layer.weight.dtype))
        low_rank.B.weight.copy_(B_weight.to(device=device, dtype=layer.weight.dtype))

    return low_rank


def compress_model_svd(
    model: nn.Module,
    rank: int,
    skip_names: tuple[str, ...] = ("lm_head", "wte", "wpe"),
) -> nn.Module:
    """Replace every nn.Linear in *model* with a LowRankLinear approximation.

    Layers whose name contains any string from *skip_names* are left intact
    (embedding / unembedding layers are typically kept full-rank).
    Modifies the model in-place and returns it.
    """
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, nn.Linear):
                continue

            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(skip in full_name for skip in skip_names):
                continue

            in_f, out_f = child_module.in_features, child_module.out_features
            effective_rank = min(rank, in_f, out_f)
            if effective_rank >= min(in_f, out_f):
                # no compression possible
                continue

            low_rank = _svd_compress_linear(child_module, effective_rank)
            setattr(parent_module, child_name, low_rank)

    return model


@torch.no_grad()
def compress_model_svd_adaptive(
    model: nn.Module,
    r_gap: int = 0,
    skip_names: tuple[str, ...] = ("lm_head", "wte", "wpe"),
) -> nn.Module:
    """Replace every nn.Linear with a LowRankLinear compressed to its own effective rank.

    The target rank for each layer is round(effective_rank(W)) + r_gap, so r_gap
    lets you compress slightly more (negative) or less (positive) than the bare
    effective rank. Modifies the model in-place and returns it.
    """
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, nn.Linear):
                continue

            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(skip in full_name for skip in skip_names):
                continue

            in_f, out_f = child_module.in_features, child_module.out_features
            max_rank = min(in_f, out_f)

            er = effective_rank(child_module.weight.data)
            target_rank = min(max(round(er) + r_gap, 1), max_rank)
            if target_rank >= max_rank:
                continue

            low_rank = _svd_compress_linear(child_module, target_rank)
            setattr(parent_module, child_name, low_rank)
            print(f"  {full_name}: erank={er:.1f} → rank={target_rank} (max={max_rank})")

    return model


@torch.no_grad()
def effective_rank(W: torch.Tensor) -> float:
    """Entropy-based effective rank (Roy & Vetterli, 2007).

    erank(W) = exp( -sum_i p_i * log(p_i) )  where p_i = s_i / sum(s).
    Ranges from 1 (rank-1 matrix) to min(m, n) (uniform singular spectrum).
    """
    s = torch.linalg.svdvals(W.float())
    s = s[s > 0]
    p = s / s.sum()
    return torch.exp(-(p * p.log()).sum()).item()


@torch.no_grad()
def model_effective_ranks(
    model: nn.Module,
    skip_names: tuple[str, ...] = ("lm_head", "wte", "wpe"),
) -> dict[str, float]:
    """Compute effective rank for every 2-D weight matrix in *model*.

    Returns a dict with:
      - ``"effective_rank/<layer_name>"`` for each qualifying layer
      - ``"effective_rank/mean_weighted"`` — weighted average across layers,
        weights = min(out, in) (number of singular values per layer)
    """
    ranks: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(skip in name for skip in skip_names):
            continue
        er = effective_rank(module.weight.data)
        ranks[f"effective_rank/{name}"] = er
        k = min(module.weight.shape)
        weighted_sum += er * k
        total_weight += k

    if total_weight > 0:
        ranks["effective_rank/mean_weighted"] = weighted_sum / total_weight

    return ranks


@torch.no_grad()
def grad_effective_ranks(
    model: nn.Module,
    skip_names: tuple[str, ...] = ("lm_head", "wte", "wpe"),
) -> dict[str, float]:
    """Weighted-mean effective rank of 2-D gradient tensors."""
    weighted_sum = 0.0
    total_weight = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(skip in name for skip in skip_names):
            continue
        if module.weight.grad is None:
            continue
        er = effective_rank(module.weight.grad)
        k = min(module.weight.shape)
        weighted_sum += er * k
        total_weight += k
    result: dict[str, float] = {}
    if total_weight > 0:
        result["grad_effective_rank/mean_weighted"] = weighted_sum / total_weight
    return result


@torch.no_grad()
def optimizer_state_effective_ranks(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    skip_names: tuple[str, ...] = ("lm_head", "wte", "wpe"),
) -> dict[str, float]:
    """Weighted-mean effective rank of 2-D optimizer state tensors.

    Auto-detects all 2-D Tensor state entries (e.g. first_momentum, second_momentum).
    """
    param_to_name = {id(p): name for name, p in model.named_parameters()}
    sums: dict[str, float] = {}
    totals: dict[str, int] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.dim() != 2:
                continue
            name = param_to_name.get(id(p), "")
            if any(skip in name for skip in skip_names):
                continue
            state = optimizer.state[p]
            if not state:
                continue
            k = min(p.shape)
            for key, val in state.items():
                if not isinstance(val, torch.Tensor) or val.dim() != 2:
                    continue
                er = effective_rank(val)
                sums[key] = sums.get(key, 0.0) + er * k
                totals[key] = totals.get(key, 0) + k
    result: dict[str, float] = {}
    for key in sums:
        if totals[key] > 0:
            result[f"opt_{key}_effective_rank/mean_weighted"] = sums[key] / totals[key]
    return result


def compression_stats(original: nn.Module, compressed: nn.Module) -> dict:
    orig_params = sum(p.numel() for p in original.parameters())
    comp_params = sum(p.numel() for p in compressed.parameters())
    return {
        "original_params": orig_params,
        "compressed_params": comp_params,
        "compression_ratio": orig_params / comp_params if comp_params > 0 else float("inf"),
    }
