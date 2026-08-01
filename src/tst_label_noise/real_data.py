"""Real UCR time series classification datasets (via aeon), as an alternative
data source to the synthetic SyntheticWaveDataset in data.py -- same
NoisyLabelSubset label-noise mechanism on top, so results are directly
comparable to §7 (synthetic waves) using real, inherently noisy sensor data
instead of a controlled synthetic signal."""
import torch
from torch.utils.data import DataLoader, Dataset

from mnist_label_noise.data import NoisyLabelSubset

from .config import TSTLabelNoiseConfig


class UCRDataset(Dataset):
    """X: (n, seq_len, 1) float tensor (already z-normalized per-series),
    y: (n,) long tensor of 0-indexed class ids."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def _load_split(dataset_name: str, split: str, label_to_idx: dict):
    from aeon.datasets import load_classification

    X, y = load_classification(dataset_name, split=split)  # X: (n, 1, seq_len)
    X = torch.tensor(X, dtype=torch.float32).squeeze(1)  # (n, seq_len)
    X = (X - X.mean(dim=1, keepdim=True)) / (X.std(dim=1, keepdim=True) + 1e-8)
    X = X.unsqueeze(-1)  # (n, seq_len, 1)
    y = torch.tensor([label_to_idx[label] for label in y], dtype=torch.long)
    return X, y


def probe_dataset_shape(dataset_name: str):
    """Returns (seq_len, num_classes) without loading the full test split,
    so the caller can size TSTLabelNoiseConfig before building the model."""
    from aeon.datasets import load_classification

    X, y = load_classification(dataset_name, split="train")
    labels = sorted(set(y.tolist()))
    return X.shape[-1], len(labels)


def get_real_loaders(cfg: TSTLabelNoiseConfig, dataset_name: str):
    from aeon.datasets import load_classification

    X_train_raw, y_train_raw = load_classification(dataset_name, split="train")
    labels = sorted(set(y_train_raw.tolist()))
    label_to_idx = {label: i for i, label in enumerate(labels)}

    x_train, y_train = _load_split(dataset_name, "train", label_to_idx)
    x_test, y_test = _load_split(dataset_name, "test", label_to_idx)

    train_full = UCRDataset(x_train, y_train)
    test_full = UCRDataset(x_test, y_test)

    idx = list(range(len(train_full)))
    train_subset = NoisyLabelSubset(train_full, idx, cfg.noise_frac, cfg.num_classes, seed=cfg.seed)

    # Eval loaders reuse cfg.batch_size (not a larger hardcoded constant): attention
    # memory scales with batch_size * seq_len^2, and long-sequence datasets like
    # StarLightCurves (seq_len=1024) OOM'd on a shared GPU with a 1024-sized eval
    # batch even though training itself (batch_size=128) was fine.
    train_loader = DataLoader(train_subset, batch_size=min(cfg.batch_size, len(train_subset)), shuffle=True)
    train_eval_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_full, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, train_eval_loader, test_loader, train_subset
