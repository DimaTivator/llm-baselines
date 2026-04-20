import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LoRAWeightHolder(nn.Module):
    """Minimal wrapper that exposes a single trainable 'weight' Parameter.

    Mirrors the _WeightHolder used in RiemannianLoraLinear so that the
    Riemannian optimizer can locate factors by the conventional names
    'lora_A.weight' / 'lora_B.weight' via param_names in the param group.
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)


class HybridLoRALinear(nn.Module):
    """
    Linear layer with a PEFT-style LoRA adapter on top.

    Both the base weight (for stateless optimizer) and the LoRA adapter
    factors lora_A / lora_B (for stateful optimizer) are kept trainable.

    Forward:
        y = x @ W.T + bias  +  (alpha/rank) * x @ A.T @ B.T

    Init convention (adapter starts at zero contribution):
        A ~ N(0, 1/sqrt(rank))
        B  = 0
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.scale = alpha / rank

        device = linear.weight.device
        dtype = linear.weight.dtype

        self.weight = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(
            torch.randn(rank, linear.in_features, device=device, dtype=dtype)
            / math.sqrt(rank)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(linear.out_features, rank, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + lora

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scale={self.scale:.4f}"
        )


class HybridRiemannianLoRALinear(nn.Module):
    """
    Linear layer with a Riemannian (Stiefel-manifold) LoRA adapter on top.

    The base weight is a regular Parameter updated by the stateless optimizer.
    The adapter factors live on the Stiefel manifold and are updated by the
    Riemannian optimizer (RiemannianLora / RiemannianSGD).

    Storage layout (matches RiemannianLoraLinear conventions so that the
    existing Riemannian optimizer works without modification):
        lora_A.weight : (2*rank, out_features)
        lora_B.weight : (in_features, 2*rank)

    Active factors extracted at forward time:
        A_active = lora_A.weight.T[:, :rank]   shape (out, rank)
        B_active = lora_B.weight.T[:rank, :]   shape (rank, in)

    Forward:
        y = W·x + bias + A_active @ B_active @ x
    """

    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank

        device = linear.weight.device
        dtype = linear.weight.dtype

        # Base weight (updated by stateless optimizer).
        self.weight = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        # lora_A.weight : (2*rank, out_features)
        # First rank rows hold A_active (orthonormal on Stiefel),
        # last rank rows are zero and are rebuilt by the retraction.
        A_init = torch.zeros(2 * rank, linear.out_features, dtype=dtype, device=device)
        Q_A = torch.linalg.qr(
            torch.randn(linear.out_features, rank, dtype=dtype, device=device)
        ).Q
        A_init[:rank, :] = Q_A.T

        # lora_B.weight : (in_features, 2*rank)
        # First rank cols hold B_active (small random init → small adapter),
        # last rank cols hold an orthonormal complement for the Stiefel structure.
        B_init = torch.zeros(linear.in_features, 2 * rank, dtype=dtype, device=device)
        target_std = math.sqrt(2.0 / linear.in_features)
        b_std = target_std * math.sqrt(linear.out_features / rank)
        B_init[:, :rank] = torch.randn(linear.in_features, rank, dtype=dtype, device=device) * b_std
        Q_B = torch.linalg.qr(
            torch.randn(linear.in_features, rank, dtype=dtype, device=device)
        ).Q
        B_init[:, rank:] = Q_B

        self.lora_A = _LoRAWeightHolder(A_init)
        self.lora_B = _LoRAWeightHolder(B_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        A_active = self.lora_A.weight.T[:, :self.rank]  # (out, rank)
        B_active = self.lora_B.weight.T[:self.rank, :]  # (rank, in)
        lora = x @ B_active.T @ A_active.T
        return base + lora

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}"
        )
