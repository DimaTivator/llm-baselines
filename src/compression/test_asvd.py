"""Test ASVD compression on a llama checkpoint.

Usage:
    python ./src/compression/test_asvd.py \
        --ckpt_path exps/llama124m_adamw_lr1e-3/ckpts/latest/main.pt \
        --rank 64 \
        --device cuda \
        --eval_batches 64 \
        --calib_batches 16

    # Use automatic per-layer effective-rank compression:
    python ./src/compression/test_asvd.py \
        --ckpt_path exps/llama124m_adamw_lr1e-3/ckpts/latest/main.pt \
        --rank auto \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression._eval_utils import (
    load_model_from_ckpt, eval_perplexity, make_calibration_dataloader,
)
from compression.asvd import collect_activation_stats, apply_asvd


def _parse_rank(value: str):
    """Accept an integer or the literal string 'auto'."""
    if value == "auto":
        return "auto"
    return int(value)


def main():
    parser = argparse.ArgumentParser(description="Test ASVD compression.")
    parser.add_argument("--ckpt_path", required=True, type=Path)
    parser.add_argument("--rank", default="64", type=_parse_rank,
                        help="Number of singular values to retain, or 'auto' "
                             "to use each layer's effective rank.")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Scaling exponent for activation-aware weighting.")
    parser.add_argument(
        "--target_modules", nargs="+",
        default=["q_proj", "v_proj", "k_proj", "c_attn", "c_proj", "c_fc"],
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--eval_batches", default=64, type=int)
    parser.add_argument("--calib_batches", default=16, type=int,
                        help="Batches used to collect activation statistics.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="ns_weights", type=str)
    parser.add_argument("--wandb_entity", default="andrey", type=str)
    args = parser.parse_args()

    model, cfg, val_reader = load_model_from_ckpt(args.ckpt_path, device=args.device)

    print("\nEvaluating original model ...")
    orig_loss, orig_ppl = eval_perplexity(model, val_reader, args.device, args.eval_batches)
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"  Original  | params={orig_params/1e6:.2f}M  loss={orig_loss:.4f}  ppl={orig_ppl:.2f}")

    # Collect activation statistics
    print(f"\nCollecting activation statistics ({args.calib_batches} batches) ...")
    calib_data = make_calibration_dataloader(val_reader, n_batches=args.calib_batches)
    activation_stats = collect_activation_stats(
        model, calib_data, n_batches=args.calib_batches, device=args.device,
    )
    print(f"  Collected stats for {len(activation_stats)} layers.")

    # Apply ASVD
    print(f"\nApplying ASVD (rank={args.rank}, alpha={args.alpha}) ...")
    _, comp_info = apply_asvd(
        model, rank=args.rank, activation_stats=activation_stats,
        alpha=args.alpha, target_modules=tuple(args.target_modules),
    )
    model.eval()

    print("Evaluating compressed model ...")
    comp_loss, comp_ppl = eval_perplexity(model, val_reader, args.device, args.eval_batches)
    print(f"  Compressed| loss={comp_loss:.4f}  ppl={comp_ppl:.2f}  Δloss={comp_loss-orig_loss:+.4f}")

    if args.wandb:
        import wandb
        exp_name = getattr(cfg, "experiment_name", "model")
        rank_tag = args.rank if args.rank == "auto" else args.rank
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{exp_name}_asvd_rank{rank_tag}",
            tags=["compress", "asvd"],
            config={"rank": rank_tag, "alpha": args.alpha,
                    "target_modules": args.target_modules,
                    "ckpt_path": str(args.ckpt_path)},
        )
        log_dict = {
            "original/loss": orig_loss, "original/ppl": orig_ppl,
            "compressed/loss": comp_loss, "compressed/ppl": comp_ppl,
        }
        if args.rank != "auto":
            log_dict["rank"] = args.rank
        wandb.log(log_dict)
        wandb.finish()


if __name__ == "__main__":
    main()
