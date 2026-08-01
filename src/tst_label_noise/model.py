from typing import Dict, Type

import torch
import torch.nn as nn

from vit_label_noise.model import Block  # generic pre-norm transformer block, not vision-specific

from .config import TSTLabelNoiseConfig

FACTORY: Dict[str, Type[nn.Module]] = {}


def register(name):
    def decorator(cls):
        FACTORY[name] = cls
        return cls

    return decorator


@register("tst")
class TimeSeriesTransformer(nn.Module):
    """Small Transformer encoder for time series classification. Same
    Linear-only building blocks as vit_label_noise.model.ViT (reused directly)
    so AdamWSpectralL1Reg / model_effective_ranks / compress_model_svd apply
    unmodified -- only the input embedding changes (per-timestep Linear instead
    of patch Linear) since there's no spatial patching for a 1D sequence."""

    def __init__(self, cfg: TSTLabelNoiseConfig, in_channels: int = 1):
        super().__init__()
        self.input_embed = nn.Linear(in_channels, cfg.dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.seq_len + 1, cfg.dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [Block(cfg.dim, cfg.heads, cfg.mlp_dim) for _ in range(cfg.depth)]
        )
        self.ln_final = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        # x: (B, seq_len, in_channels)
        x = self.input_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.head(x[:, 0])


def get_model(name: str, cfg: TSTLabelNoiseConfig) -> nn.Module:
    if name not in FACTORY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(FACTORY)}")
    return FACTORY[name](cfg)
