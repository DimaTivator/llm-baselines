#!/usr/bin/env python3
"""Compute mean weighted stable rank for a sweep of model checkpoints.

The layer selection and weighting match ``model_effective_ranks``: token
embeddings and the LM head are skipped, and each Linear/Conv2d layer is weighted
by its number of singular values, ``min(out_dim, in_dim)``.

Example:
  PYTHONPATH=./src python src/stable_rank_analyze.py \
      --exp_root exps/cf_bruteforce_124M \
      --model_prefix llama124m \
      --device cuda:0 \
      --output_dir results/stable_rank_124m
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from compression._eval_utils import load_config
from models.compress import model_stable_ranks
from models.utils import get_model


DEFAULT_COEFS = (0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0)


def _checkpoint_for_coef(exp_root: Path, model_prefix: str, coef: float) -> Path:
    coef_label = f"{coef:g}"
    pattern = f"{model_prefix}_*_sl1_{coef_label}_finewebedu_erank"
    matches = sorted(path for path in exp_root.glob(pattern) if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one experiment matching {pattern!r}, found {len(matches)}: "
            f"{[path.name for path in matches]}"
        )
    checkpoint = matches[0] / "ckpts" / "latest" / "main.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def _load_model(checkpoint: Path, device: str) -> torch.nn.Module:
    cfg = load_config(checkpoint)
    cfg.use_pretrained = "none"
    model = get_model(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    del state
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--model_prefix", default="llama124m")
    parser.add_argument("--coefs", type=float, nargs="+", default=DEFAULT_COEFS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for coef in args.coefs:
        checkpoint = _checkpoint_for_coef(args.exp_root, args.model_prefix, coef)
        print(f"[coef={coef:g}] loading {checkpoint}", flush=True)
        model = _load_model(checkpoint, args.device)
        ranks = model_stable_ranks(model)
        mean_weighted = ranks["stable_rank/mean_weighted"]
        row = {
            "coef": coef,
            "checkpoint": str(checkpoint),
            "mean_weighted_stable_rank": mean_weighted,
            "n_layers": len(ranks) - 1,
            "stable_ranks": ranks,
        }
        rows.append(row)
        print(
            f"[coef={coef:g}] mean_weighted_stable_rank={mean_weighted:.6f} "
            f"layers={row['n_layers']}",
            flush=True,
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        payload = {
            "exp_root": str(args.exp_root),
            "model_prefix": args.model_prefix,
            "weighting": "min(out_dim, in_dim)",
            "skip_names": ["lm_head", "wte", "wpe"],
            "rows": rows,
        }
        (args.output_dir / "stable_ranks.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )

    markdown = [
        "| spectral L1 coef | mean weighted stable rank | layers |",
        "| ---: | ---: | ---: |",
    ]
    markdown.extend(
        f"| {row['coef']:g} | {row['mean_weighted_stable_rank']:.6f} | {row['n_layers']} |"
        for row in rows
    )
    (args.output_dir / "stable_ranks.md").write_text("\n".join(markdown) + "\n")


if __name__ == "__main__":
    main()
