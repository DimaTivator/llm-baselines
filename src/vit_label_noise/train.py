"""Label-noise generalization study on CIFAR10 with a ViT, mirroring
mnist_label_noise/train.py. Reuses AdamWSpectralL1Reg (optim/adamw_spectral_L1_reg.py)
and the effective-rank / truncated-SVD utilities (models/compress.py) unmodified --
the ViT here (model.py) is built entirely out of nn.Linear (Linear patch embed,
manual Linear-only attention), so nothing about those utilities needed to change.

Must be run with `src/` on sys.path, e.g. from the repo root:
    PYTHONPATH=./src python -m vit_label_noise.train --noise_frac 0.25 --spectral_l1_reg_coef 1.0
See scripts/vit_label_noise/train_noisefrac_sweep.sh for the full sweep.
"""

import argparse
import copy
import csv
import dataclasses
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from models.compress import compress_model_svd, model_effective_ranks
from optim.adamw_spectral_L1_reg import AdamWSpectralL1Reg

from .config import ViTLabelNoiseConfig
from .data import get_loaders
from .model import get_model

logger = logging.getLogger(__name__)

RANK_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        n += y.size(0)
    model.train()
    return total_loss / n, correct / n


@torch.no_grad()
def clean_vs_noisy_train_acc(model, train_subset, device):
    model.eval()
    loader = torch.utils.data.DataLoader(train_subset, batch_size=1024, shuffle=False)
    correct_clean, n_clean, correct_noisy, n_noisy = 0, 0, 0, 0
    offset = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=-1)
        batch_is_noisy = train_subset.is_noisy[offset: offset + y.size(0)]
        offset += y.size(0)
        correct = pred == y
        correct_clean += correct[~batch_is_noisy].sum().item()
        n_clean += (~batch_is_noisy).sum().item()
        correct_noisy += correct[batch_is_noisy].sum().item()
        n_noisy += batch_is_noisy.sum().item()
    model.train()
    return correct_clean / max(n_clean, 1), correct_noisy / max(n_noisy, 1)


def warmup_cosine_lambda(total_steps: int, warmup_steps: int):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return fn


@torch.no_grad()
def truncation_sweep(model, test_loader, device):
    curve = []
    for rank in RANK_GRID:
        compressed = compress_model_svd(copy.deepcopy(model), rank=rank, skip_names=()).to(device)
        _, acc = evaluate(compressed, test_loader, device)
        curve.append({"rank": rank, "test_acc": acc})
    return curve


def train_one_run(cfg: ViTLabelNoiseConfig) -> dict:
    torch.manual_seed(cfg.seed)
    run_name = cfg.run_name or f"vit_d{cfg.dim}x{cfg.depth}_nf{cfg.noise_frac}_sl1_{cfg.spectral_l1_reg_coef}_seed{cfg.seed}"

    if cfg.wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=run_name, config=dataclasses.asdict(cfg))

    train_loader, train_eval_loader, test_loader, train_subset = get_loaders(cfg)

    model = get_model("vit", cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("run=%s params=%.2fM", run_name, n_params / 1e6)

    optimizer = AdamWSpectralL1Reg(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        spectral_l1_reg_coef=cfg.spectral_l1_reg_coef,
        svt_interval=0,
    )
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_frac * total_steps)
    scheduler = LambdaLR(optimizer, warmup_cosine_lambda(total_steps, warmup_steps))

    history = []
    for epoch in range(cfg.epochs):
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss, train_acc = evaluate(model, train_eval_loader, cfg.device)
        test_loss, test_acc = evaluate(model, test_loader, cfg.device)
        clean_acc_ep, noisy_acc_ep = clean_vs_noisy_train_acc(model, train_subset, cfg.device)
        row = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
            "clean_label_train_acc": clean_acc_ep, "corrupted_label_train_acc": noisy_acc_ep,
        }
        history.append(row)
        if cfg.wandb:
            wandb.log(row, step=epoch)
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            logger.info(
                "run=%s epoch=%d train_acc=%.4f test_acc=%.4f corrupted_train_acc=%.4f",
                run_name, epoch, train_acc, test_acc, noisy_acc_ep,
            )

    # torch.linalg.svd is not implemented on MPS; do the SVD-based analysis on CPU.
    cpu_model = copy.deepcopy(model).to("cpu")
    ranks = model_effective_ranks(cpu_model, skip_names=())
    curve = truncation_sweep(cpu_model, test_loader, "cpu")
    final = history[-1]
    result = {
        "run_name": run_name,
        "seed": cfg.seed,
        "spectral_l1_reg_coef": cfg.spectral_l1_reg_coef,
        "noise_frac": cfg.noise_frac,
        "n_params": n_params,
        "final_train_acc": final["train_acc"],
        "final_test_acc": final["test_acc"],
        "clean_label_train_acc": final["clean_label_train_acc"],
        "corrupted_label_train_acc": final["corrupted_label_train_acc"],
        "effective_ranks": ranks,
        "history": history,
        "truncation_curve": curve,
    }

    if cfg.wandb:
        import wandb
        wandb.log({"effective_ranks": ranks, "truncation_curve": curve})
        wandb.finish()

    run_dir = Path(cfg.results_base_folder) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    logger.info("saved results to %s", run_dir)

    return result


def parse_args() -> ViTLabelNoiseConfig:
    p = argparse.ArgumentParser()
    defaults = dataclasses.asdict(ViTLabelNoiseConfig())
    for field in dataclasses.fields(ViTLabelNoiseConfig):
        default = defaults[field.name]
        if field.type == bool:
            p.add_argument(f"--{field.name}", action="store_true", default=default)
        else:
            p.add_argument(f"--{field.name}", type=type(default) if default is not None else str, default=default)
    args = p.parse_args()
    return ViTLabelNoiseConfig(**vars(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = parse_args()
    train_one_run(cfg)
