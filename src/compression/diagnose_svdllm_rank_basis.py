#!/usr/bin/env python3
"""Diagnose the rank basis and numerical conditioning of SVD-LLM layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compression._eval_utils import load_model_from_ckpt, make_calibration_dataloader
from compression.svd_llm import collect_input_covariance
from data.utils import DataReader, get_dataset
from models.compress import effective_rank


TARGET_MODULES = ("c_attn", "c_proj", "w1", "w2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calib_batches", type=int, default=16)
    parser.add_argument("--calib_batch_size", type=int, default=8)
    parser.add_argument("--auto_rank_multiple", type=int, default=16)
    return parser.parse_args()


def aligned_rank(rank: float, max_rank: int, multiple: int) -> int:
    return min(max(16, (round(rank) // multiple) * multiple), max_rank)


def build_val_reader(cfg, batch_size: int) -> DataReader:
    return DataReader(
        data_src=get_dataset(cfg)["val"],
        batch_size=batch_size,
        sequence_length=cfg.sequence_length,
        seed=cfg.data_seed,
        with_replacement=False,
        auto_shard=False,
    )


def main() -> None:
    args = parse_args()
    if args.auto_rank_multiple < 16 or args.auto_rank_multiple % 16:
        raise ValueError("auto_rank_multiple must be a positive multiple of 16")

    model, cfg, _ = load_model_from_ckpt(args.ckpt_path, device=args.device)
    reader = build_val_reader(cfg, args.calib_batch_size)
    calibration_data = make_calibration_dataloader(reader, args.calib_batches)
    whitening = collect_input_covariance(
        model,
        calibration_data,
        n_batches=len(calibration_data),
        device=args.device,
    )

    rows: list[dict] = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not any(name.endswith(suffix) for suffix in TARGET_MODULES):
                continue
            key = f"{name}.weight"
            if key not in whitening:
                raise KeyError(f"Missing covariance for {key}")

            weight = module.weight.detach().float()
            max_rank = min(weight.shape)
            covariance = whitening[key].float().to(weight.device)
            eigenvalues, eigenvectors = torch.linalg.eigh(
                covariance
                + 1e-6 * torch.eye(covariance.shape[0], device=weight.device)
            )
            eigenvalues = eigenvalues.clamp_min(0.0)
            sqrt_covariance = (
                eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T
            )
            inv_sqrt_covariance = (
                eigenvectors
                @ torch.diag(1.0 / (eigenvalues.sqrt() + 1e-8))
                @ eigenvectors.T
            )
            whitened_weight = weight @ sqrt_covariance
            _, singular_values, vh = torch.linalg.svd(
                whitened_weight, full_matrices=False
            )

            weight_rank = aligned_rank(
                effective_rank(weight), max_rank, args.auto_rank_multiple
            )
            whitened_rank = aligned_rank(
                effective_rank(whitened_weight),
                max_rank,
                args.auto_rank_multiple,
            )
            whitened_norm_sq = singular_values.square().sum()

            def residual(rank: int) -> float:
                return float(
                    (singular_values[rank:].square().sum() / whitened_norm_sq)
                    .sqrt()
                    .item()
                )

            right_factor = vh[:weight_rank] @ inv_sqrt_covariance
            rows.append(
                {
                    "layer": name,
                    "shape": list(weight.shape),
                    "weight_erank": effective_rank(weight),
                    "whitened_erank": effective_rank(whitened_weight),
                    "weight_rank": weight_rank,
                    "whitened_rank": whitened_rank,
                    "rank_delta": whitened_rank - weight_rank,
                    "weight_rank_whitened_relative_residual": residual(weight_rank),
                    "whitened_rank_whitened_relative_residual": residual(
                        whitened_rank
                    ),
                    "covariance_condition": float(
                        (eigenvalues.max() / eigenvalues.min()).item()
                    ),
                    "right_factor_abs_max": float(right_factor.abs().max().item()),
                    "right_factor_frobenius_norm": float(right_factor.norm().item()),
                }
            )
            print(
                f"{name}: rank W={weight_rank}, WS={whitened_rank}, "
                f"residual={residual(weight_rank):.5f}",
                flush=True,
            )

    payload = {
        "checkpoint": str(args.ckpt_path),
        "calib_batches": args.calib_batches,
        "calib_batch_size": args.calib_batch_size,
        "auto_rank_multiple": args.auto_rank_multiple,
        "rows": rows,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_path.with_suffix(args.output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output_path)
    print(f"RESULT={args.output_path}", flush=True)


if __name__ == "__main__":
    main()
