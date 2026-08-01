from typing import Dict, Type

import torch.nn as nn

from .config import CNNLabelNoiseConfig

FACTORY: Dict[str, Type[nn.Module]] = {}


def register(name):
    def decorator(cls):
        FACTORY[name] = cls
        return cls

    return decorator


def conv_block(in_ch: int, out_ch: int, pool: bool) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


@register("cnn")
class SimpleCNN(nn.Module):
    """Small CNN for CIFAR10: 3 stages of 2 conv blocks each (32x32 -> 16x16 ->
    8x8 -> 4x4), global average pool, Linear head. Conv2d weights are 4D
    [out_ch, in_ch, kh, kw] -- AdamWSpectralL1Reg reshapes them to the 2D
    filter matrix [out_ch, in_ch*kh*kw] before applying the nuclear-norm prox
    (see optim/adamw_spectral_L1_reg.py), so spectral WD applies to every conv
    layer here unmodified. The classifier head is kept out of the regularized
    param group in train.py (known small-head rank-collapse risk from the
    time-series studies), not architecturally excluded here.
    """

    def __init__(self, cfg: CNNLabelNoiseConfig, in_channels: int = 3, out_dim: int = 10):
        super().__init__()
        c = cfg.base_channels
        self.features = nn.Sequential(
            conv_block(in_channels, c, pool=False),
            conv_block(c, c, pool=True),
            conv_block(c, 2 * c, pool=False),
            conv_block(2 * c, 2 * c, pool=True),
            conv_block(2 * c, 4 * c, pool=False),
            conv_block(4 * c, 4 * c, pool=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4 * c, out_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def get_model(name: str, cfg: CNNLabelNoiseConfig) -> nn.Module:
    if name not in FACTORY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(FACTORY)}")
    return FACTORY[name](cfg)
