import math

import torch
from torch.utils.data import DataLoader, Dataset

from mnist_label_noise.data import NoisyLabelSubset

from .config import TSTLabelNoiseConfig


class SyntheticWaveDataset(Dataset):
    """Class-conditional synthetic time series: class k has its own base
    frequency/phase/amplitude distribution (a two-harmonic sine mixture), plus
    i.i.d. Gaussian observation noise on top -- the "noisy time series" itself,
    separate from the label-noise mechanism applied on top via NoisyLabelSubset.
    Pregenerated (not streamed) since dataset sizes here are small."""

    def __init__(self, n: int, seq_len: int, num_classes: int, obs_noise_std: float, seed: int):
        g = torch.Generator().manual_seed(seed)
        t = torch.linspace(0, 1, seq_len)

        labels = torch.randint(0, num_classes, (n,), generator=g)
        # base frequency per class, spread out so classes are separable in the noiseless limit
        class_freq = 2.0 + 1.5 * torch.arange(num_classes)
        phase = torch.rand(n, generator=g) * 2 * math.pi
        amp2 = 0.3 + 0.2 * torch.rand(n, generator=g)

        freq = class_freq[labels]
        x = torch.sin(2 * math.pi * freq.unsqueeze(1) * t.unsqueeze(0) + phase.unsqueeze(1))
        x = x + amp2.unsqueeze(1) * torch.sin(2 * 2 * math.pi * freq.unsqueeze(1) * t.unsqueeze(0))
        x = x + obs_noise_std * torch.randn(n, seq_len, generator=g)

        self.x = x.unsqueeze(-1)  # (n, seq_len, 1) -- single input channel
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def get_loaders(cfg: TSTLabelNoiseConfig):
    train_full = SyntheticWaveDataset(
        n=cfg.train_subset_size, seq_len=cfg.seq_len, num_classes=cfg.num_classes,
        obs_noise_std=cfg.obs_noise_std, seed=cfg.seed,
    )
    test_full = SyntheticWaveDataset(
        n=cfg.test_size, seq_len=cfg.seq_len, num_classes=cfg.num_classes,
        obs_noise_std=cfg.obs_noise_std, seed=cfg.seed + 10_000,  # disjoint stream from train
    )

    idx = list(range(len(train_full)))
    train_subset = NoisyLabelSubset(train_full, idx, cfg.noise_frac, cfg.num_classes, seed=cfg.seed)

    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_subset, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_full, batch_size=1024, shuffle=False)
    return train_loader, train_eval_loader, test_loader, train_subset
