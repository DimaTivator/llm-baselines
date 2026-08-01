from typing import Dict, Type

import torch
import torch.nn as nn

from .config import ViTLabelNoiseConfig

FACTORY: Dict[str, Type[nn.Module]] = {}


def register(name):
    def decorator(cls):
        FACTORY[name] = cls
        return cls

    return decorator


class Attention(nn.Module):
    """Multi-head self-attention with every weight matrix a plain 2D nn.Linear
    (packed qkv projection, like LlamaAttention.c_attn in models/llama.py) so
    AdamWSpectralL1Reg's nuclear-norm prox applies to it directly -- no conv-style
    reshaping needed, unlike nn.MultiheadAttention's raw in_proj_weight."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.c_attn = nn.Linear(dim, dim * 3, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.c_attn(x).split(D, dim=2)
        q = q.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.c_proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(dim, mlp_dim, bias=False)
        self.c_proj = nn.Linear(mlp_dim, dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@register("vit")
class ViT(nn.Module):
    """Vision Transformer with a Linear (not Conv2d) patch embedding and manual
    (Linear-only) attention, so every weight matrix that matters is 2D -- spectral
    WD (AdamWSpectralL1Reg), model_effective_ranks, and compress_model_svd in
    models/compress.py all apply to it unmodified, same as the MLP label-noise
    study. cls token / position embeddings / LayerNorm affine params are left
    untouched by the nuclear-norm prox (not 2D), same as biases elsewhere."""

    def __init__(self, cfg: ViTLabelNoiseConfig, in_channels: int = 3, out_dim: int = 10):
        super().__init__()
        assert cfg.image_size % cfg.patch_size == 0
        n_patches = (cfg.image_size // cfg.patch_size) ** 2
        patch_dim = in_channels * cfg.patch_size ** 2

        self.patch_size = cfg.patch_size
        self.patch_embed = nn.Linear(patch_dim, cfg.dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, cfg.dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [Block(cfg.dim, cfg.heads, cfg.mlp_dim) for _ in range(cfg.depth)]
        )
        self.ln_final = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, out_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Standard ViT init (e.g. DeiT/timm): trunc_normal_(std=0.02) for Linear
        # weights, zero bias -- PyTorch's default (kaiming-uniform-ish) init is
        # not calibrated for this depth and, combined with no LR warmup, made
        # training unstable/slow (observed as near-chance test acc even at
        # noise_frac=0.1, where the MLP-on-MNIST study converges cleanly).
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _to_patches(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)  # B, C, H/p, W/p, p, p
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)  # B, n_patches, patch_dim
        return x

    def forward(self, x):
        x = self._to_patches(x)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.head(x[:, 0])


def get_model(name: str, cfg: ViTLabelNoiseConfig) -> nn.Module:
    if name not in FACTORY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(FACTORY)}")
    return FACTORY[name](cfg)
