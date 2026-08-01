"""Test FWSVD compression on a llama checkpoint.

Usage:
    python ./src/compression/test_fwsvd.py \
        --ckpt_path exps/llama124m_adamw_lr1e-3/ckpts/latest/main.pt \
        --rank 64 \
        --device cuda \
        --eval_batches 64 \
        --calib_batches 16
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression._eval_utils import (
    load_model_from_ckpt, eval_perplexity, make_calibration_dataloader,
)
from compression.fwsvd import collect_fisher_stats, apply_fwsvd


def main():
    parser = argparse.ArgumentParser(description="Test FWSVD compression.")
    parser.add_argument("--ckpt_path", required=True, type=Path)
    parser.add_argument("--rank", default=64, type=int)
    parser.add_argument(
        "--target_modules", nargs="+",
        default=["q_proj", "v_proj", "k_proj", "c_attn", "c_proj", "c_fc"],
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--eval_batches", default=64, type=int)
    parser.add_argument("--calib_batches", default=16, type=int,
                        help="Batches to compute Fisher (gradient) statistics.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="ns_weights", type=str)
    parser.add_argument("--wandb_entity", default="andrey", type=str)
    args = parser.parse_args()

    model, cfg, val_reader = load_model_from_ckpt(args.ckpt_path, device=args.device)

    print("\nEvaluating original model ...")
    orig_loss, orig_ppl = eval_perplexity(model, val_reader, args.device, args.eval_batches)
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"  Original  | params={orig_params/1e6:.2f}M  loss={orig_loss:.4f}  ppl={orig_ppl:.2f}")

    # Collect Fisher statistics (requires gradients)
    print(f"\nCollecting Fisher statistics ({args.calib_batches} batches) ...")
    calib_data = make_calibration_dataloader(val_reader, n_batches=args.calib_batches)
    fisher_stats = collect_fisher_stats(
        model, calib_data, n_batches=args.calib_batches, device=args.device,
    )
    print(f"  Collected Fisher stats for {len(fisher_stats)} layers.")
    model.eval()

    # Apply FWSVD
    print(f"\nApplying FWSVD (rank={args.rank}) ...")
    apply_fwsvd(
        model, rank=args.rank, fisher_stats=fisher_stats,
        target_modules=tuple(args.target_modules),
    )
    model.eval()

    print("Evaluating compressed model ...")
    comp_loss, comp_ppl = eval_perplexity(model, val_reader, args.device, args.eval_batches)
    print(f"  Compressed| loss={comp_loss:.4f}  ppl={comp_ppl:.2f}  Δloss={comp_loss-orig_loss:+.4f}")

    if args.wandb:
        import wandb
        exp_name = getattr(cfg, "experiment_name", "model")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{exp_name}_fwsvd_rank{args.rank}",
            tags=["compress", "fwsvd"],
            config={"rank": args.rank, "target_modules": args.target_modules,
                    "ckpt_path": str(args.ckpt_path)},
        )
        wandb.log({
            "original/loss": orig_loss, "original/ppl": orig_ppl,
            "compressed/loss": comp_loss, "compressed/ppl": comp_ppl,
            "rank": args.rank,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
