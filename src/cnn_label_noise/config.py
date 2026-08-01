import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CNNLabelNoiseConfig:
    seed: int = 0
    spectral_l1_reg_coef: float = 0.0
    noise_frac: float = 0.25
    # Default (False) matches §10 of the report: head is a separate optimizer
    # param group with spectral_l1_reg_coef=0, never regularized. Set True to
    # put the head in the same regularized group as everything else, to test
    # whether the head (not the conv backbone) is the mechanism's bottleneck.
    include_head_in_reg: bool = False

    base_channels: int = 64  # channel counts per block: base, base, 2*base, 2*base, 4*base, 4*base

    train_subset_size: int = 8000
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    warmup_frac: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    device: str = "cpu"

    wandb: bool = False
    wandb_project: str = "ns_weights"
    wandb_entity: str = "andrey"
    run_name: Optional[str] = None

    # Same convention as vit_label_noise/config.py: CIFAR10 must never be
    # downloaded into the shared DATASETS_DIR used elsewhere in this repo.
    datasets_dir: str = field(default_factory=lambda: os.path.expanduser("~/cifar10_datasets"))
    results_base_folder: str = "./exps/cnn_label_noise"
