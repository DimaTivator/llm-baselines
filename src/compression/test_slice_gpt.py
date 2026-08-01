"""Test SliceGPT (simplified) compression on a llama checkpoint.

Usage:
    python ./src/compression/test_slice_gpt.py \
        --ckpt_path exps/llama124m_adamw_lr1e-3/ckpts/latest/main.pt \
        --slice_fraction 0.1 \
        --device cuda \
        --eval_batches 64
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression._eval_utils import load_model_from_ckpt, eval_perplexity
from compression.slice_gpt import apply_slice_gpt


def main():
    parser = argparse.ArgumentParser(description="Test SliceGPT (simplified) compression.")
    parser.add_argument("--ckpt_path", required=True, type=Path)
    parser.add_argument("--slice_fraction", default=0.1, type=float,
                        help="Fraction of singular values to remove (0.1 = 10%%).")
    # --rank kept for API consistency with other test scripts; overrides slice_fraction
    parser.add_argument("--rank", default=None, type=int,
                        help="If set, use fixed rank instead of slice_fraction.")
    parser.add_argument(
        "--target_modules", nargs="+",
        default=["q_proj", "v_proj", "k_proj", "c_attn", "c_proj", "c_fc"],
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--eval_batches", default=64, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="ns_weights", type=str)
    parser.add_argument("--wandb_entity", default="andrey", type=str)
    args = parser.parse_args()

    model, cfg, val_reader = load_model_from_ckpt(args.ckpt_path, device=args.device)

    # Optionally derive slice_fraction from --rank
    slice_fraction = args.slice_fraction
    if args.rank is not None:
        # Approximate slice_fraction from rank relative to model hidden size
        d = getattr(cfg, "n_embd", 768)
        slice_fraction = max(0.0, 1.0 - args.rank / d)
        print(f"  Derived slice_fraction={slice_fraction:.4f} from rank={args.rank}, n_embd={d}")

    print("\nEvaluating original model ...")
    orig_loss, orig_ppl = eval_perplexity(model, val_reader, args.device, args.eval_batches)
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"  Original  | params={orig_params/1e6:.2f}M  loss={orig_loss:.4f}  ppl={orig_ppl:.2f}")

    print(f"\nApplying SliceGPT-simplified (slice_fraction={slice_fraction:.3f}) ...")
    apply_slice_gpt(
        model,
        slice_fraction=slice_fraction,
        target_modules=tuple(args.target_modules),
        device=args.device,
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
            name=f"{exp_name}_slice_gpt_sf{slice_fraction:.2f}",
            tags=["compress", "slice_gpt"],
            config={"slice_fraction": slice_fraction,
                    "target_modules": args.target_modules,
                    "ckpt_path": str(args.ckpt_path)},
        )
        wandb.log({
            "original/loss": orig_loss, "original/ppl": orig_ppl,
            "compressed/loss": comp_loss, "compressed/ppl": comp_ppl,
            "slice_fraction": slice_fraction,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
