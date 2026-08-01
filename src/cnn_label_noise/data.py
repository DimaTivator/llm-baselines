import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_label_noise.data import NoisyLabelSubset

from .config import CNNLabelNoiseConfig

NUM_CLASSES = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_loaders(cfg: CNNLabelNoiseConfig):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train_full = datasets.CIFAR10(root=cfg.datasets_dir, train=True, download=True, transform=tfm)
    test_full = datasets.CIFAR10(root=cfg.datasets_dir, train=False, download=True, transform=tfm)

    g = torch.Generator().manual_seed(cfg.seed)
    idx = torch.randperm(len(train_full), generator=g)[: cfg.train_subset_size].tolist()
    train_subset = NoisyLabelSubset(train_full, idx, cfg.noise_frac, NUM_CLASSES, seed=cfg.seed)

    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_full, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, train_eval_loader, test_loader, train_subset
