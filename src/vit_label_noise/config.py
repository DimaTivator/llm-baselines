import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ViTLabelNoiseConfig:
    seed: int = 0
    spectral_l1_reg_coef: float = 0.0
    noise_frac: float = 0.25

    image_size: int = 32
    patch_size: int = 4
    dim: int = 384
    depth: int = 8
    heads: int = 6
    mlp_dim: int = 1536

    train_subset_size: int = 8000
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    # Fraction of total steps spent on linear LR warmup before cosine decay.
    # ViTs trained from scratch are known to be unstable/slow without this.
    warmup_frac: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    device: str = "cpu"

    wandb: bool = False
    wandb_project: str = "ns_weights"
    wandb_entity: str = "andrey"
    run_name: Optional[str] = None

    # Deliberately not "./datasets" or reading a DATASETS_DIR env var: that name/path
    # is used elsewhere in this repo for the shared fineweb-edu corpus. Home directory
    # is always writable, mirrors mnist_label_noise/config.py's convention.
    datasets_dir: str = field(default_factory=lambda: os.path.expanduser("~/cifar10_datasets"))
    results_base_folder: str = "./exps/vit_label_noise"
