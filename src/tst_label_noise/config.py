from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TSTLabelNoiseConfig:
    seed: int = 0
    spectral_l1_reg_coef: float = 0.0
    noise_frac: float = 0.25

    seq_len: int = 100
    num_classes: int = 5
    obs_noise_std: float = 0.3  # additive Gaussian noise on the synthetic waveform itself

    dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_dim: int = 256

    train_subset_size: int = 2000
    test_size: int = 2000
    batch_size: int = 128
    epochs: int = 60
    lr: float = 1e-3
    warmup_frac: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    device: str = "cpu"

    wandb: bool = False
    wandb_project: str = "ns_weights"
    wandb_entity: str = "andrey"
    run_name: Optional[str] = None

    results_base_folder: str = "./exps/tst_label_noise"
