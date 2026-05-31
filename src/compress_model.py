"""Compress a trained model checkpoint via truncated SVD and evaluate.

Usage:
    python src/compress_model.py \
        --ckpt_path exps/<exp_name>/ckpts/latest/main.pt \
        --svd_rank 64 \
        [--output_path exps/<exp_name>/compressed/rank_64/main.pt] \
        [--device cpu] \
        [--eval_batches 64] \
        [--wandb] [--wandb_project my-project] [--wandb_entity my-entity]

The script reads the model architecture from summary.json located in the
experiment directory (two levels up from the checkpoint file by convention).
"""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb

# Allow importing project modules from src/
sys.path.insert(0, str(Path(__file__).parent))

from data.utils import DataReader, get_dataset
from models.compress import compress_model_svd, compress_model_svd_adaptive, model_effective_ranks
from models.utils import get_model
from optim.utils import eval as eval_model


def load_config(ckpt_path: Path) -> SimpleNamespace:
    """Walk up from the checkpoint to find summary.json with training args."""
    for parent in ckpt_path.parents:
        candidate = parent / "summary.json"
        if candidate.exists():
            with open(candidate) as f:
                args_dict = json.load(f)["args"]
            return SimpleNamespace(**args_dict)
    raise FileNotFoundError(
        f"Could not find summary.json in any parent of {ckpt_path}. "
        "Place summary.json in the experiment directory."
    )


def main():
    parser = argparse.ArgumentParser(description="Compress a model checkpoint via truncated SVD.")
    parser.add_argument("--ckpt_path", required=True, type=Path, help="Path to main.pt checkpoint.")
    parser.add_argument("--svd_rank", default=None, type=int, help="Rank for truncated SVD (fixed). Omit when using --adaptive.")
    parser.add_argument("--adaptive", action="store_true", help="Compress each layer to its own effective rank instead of a fixed rank.")
    parser.add_argument("--r_gap", default=0, type=int, help="Integer added to each layer's effective rank when using --adaptive (can be negative).")
    parser.add_argument(
        "--output_path", default=None, type=Path,
        help="Where to save the compressed model. Defaults to <ckpt_dir>/compressed/rank_<N>/main.pt.",
    )
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument(
        "--skip_names", nargs="*", default=["lm_head", "wte", "wpe"],
        help="Layer name substrings to skip during compression.",
    )
    parser.add_argument("--no_save", action="store_true", help="Skip saving the compressed model.")
    parser.add_argument("--eval_batches", default=64, type=int,
                        help="Number of batches for evaluation.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="compression", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    args = parser.parse_args()

    if not args.adaptive and args.svd_rank is None:
        parser.error("Provide --svd_rank or --adaptive.")

    ckpt_path: Path = args.ckpt_path.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("Loading config from summary.json ...")
    cfg = load_config(ckpt_path)

    if args.adaptive:
        run_label = f"adaptive_gap{args.r_gap:+d}" if args.r_gap != 0 else "adaptive"
    else:
        run_label = f"rank{args.svd_rank}"
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{getattr(cfg, 'experiment_name', 'model')}_svd_{run_label}",
            tags=["compress"],
            config={
                "svd_rank": args.svd_rank,
                "adaptive": args.adaptive,
                "skip_names": args.skip_names,
                "ckpt_path": str(ckpt_path),
                "original_experiment": getattr(cfg, "experiment_name", None),
            },
        )

    print(f"Building model ({cfg.model}) ...")
    cfg.use_pretrained = "none"
    model = get_model(cfg).to(args.device)

    print(f"Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- Evaluate original model ---
    print("Loading validation data ...")
    val_reader = DataReader(
        data_src=get_dataset(cfg)["val"],
        batch_size=cfg.batch_size,
        sequence_length=cfg.sequence_length,
        seed=cfg.data_seed,
        with_replacement=False,
        auto_shard=False,
    )

    type_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if "cuda" in args.device else nullcontext()
    )

    print("Evaluating original model ...")
    _, orig_loss, orig_ppl, _, _ = eval_model(
        model, val_reader, device=args.device,
        max_num_batches=args.eval_batches, ctx=type_ctx,
    )
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"  params={orig_params/1e6:.2f}M  loss={orig_loss:.4f}  ppl={orig_ppl:.2f}")

    # --- Compress ---
    if args.adaptive:
        print(f"Compressing with adaptive SVD (per-layer effective rank, r_gap={args.r_gap}) ...")
        compress_model_svd_adaptive(model, r_gap=args.r_gap, skip_names=tuple(args.skip_names))
    else:
        print(f"Compressing with SVD rank={args.svd_rank} ...")
        compress_model_svd(model, rank=args.svd_rank, skip_names=tuple(args.skip_names))
    model.eval()

    comp_params = sum(p.numel() for p in model.parameters())
    ratio = orig_params / comp_params
    print(
        f"  compressed params={comp_params/1e6:.2f}M  "
        f"ratio={ratio:.2f}x  saved={(1 - 1/ratio)*100:.1f}%"
    )

    # --- Evaluate compressed model ---
    print("Evaluating compressed model ...")
    _, comp_loss, comp_ppl, _, _ = eval_model(
        model, val_reader, device=args.device,
        max_num_batches=args.eval_batches, ctx=type_ctx,
    )
    loss_delta = comp_loss - orig_loss
    print(f"  loss={comp_loss:.4f}  ppl={comp_ppl:.2f}  Δloss={loss_delta:+.4f}")

    # --- Effective ranks of compressed model ---
    er_logs = model_effective_ranks(model)
    print(f"  mean weighted effective rank: {er_logs.get('effective_rank/mean_weighted', 'N/A'):.2f}")

    # --- Log to wandb ---
    if args.wandb:
        wandb.log({
            "original/loss": orig_loss,
            "compressed/loss": comp_loss,
            "svd_rank": args.svd_rank,
        })
        wandb.finish()

    # --- Save ---
    if not args.no_save:
        if args.output_path is not None:
            out_path: Path = args.output_path
        else:
            out_path = ckpt_path.parent / "compressed" / run_label / "main.pt"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "svd_rank": args.svd_rank}, out_path)
        print(f"Saved compressed model to {out_path}")


if __name__ == "__main__":
    main()
