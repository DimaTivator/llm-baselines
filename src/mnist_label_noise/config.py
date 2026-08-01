import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MLPLabelNoiseConfig:
    seed: int = 0
    spectral_l1_reg_coef: float = 0.0
    noise_frac: float = 0.25

    hidden_dim: int = 2048
    n_hidden_layers: int = 4

    train_subset_size: int = 3000
    batch_size: int = 128
    epochs: int = 60
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95

    device: str = "cpu"

    wandb: bool = False
    wandb_project: str = "ns_weights"
    wandb_entity: str = "andrey"
    run_name: Optional[str] = None

    # Deliberately not "./datasets" or reading a DATASETS_DIR env var: that name/path
    # is used elsewhere in this repo for the shared fineweb-edu corpus, and MNIST must
    # never be downloaded there (often read-only/shared). Home directory is always writable.
    datasets_dir: str = field(default_factory=lambda: os.path.expanduser("~/mnist_datasets"))
    results_base_folder: str = "./exps/mnist_label_noise"
