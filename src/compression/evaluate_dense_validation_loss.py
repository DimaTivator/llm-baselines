#!/usr/bin/env python3
"""Evaluate a dense checkpoint on the SVD-LLM validation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from compression._eval_utils import eval_perplexity, load_model_from_ckpt
from data.utils import DataReader, get_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    return parser.parse_args()


def build_val_reader(cfg: SimpleNamespace, batch_size: int) -> DataReader:
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
    if not args.ckpt_path.is_file():
        raise FileNotFoundError(args.ckpt_path)

    model, cfg, _ = load_model_from_ckpt(args.ckpt_path, device=args.device)
    reader = build_val_reader(cfg, args.eval_batch_size)
    print("Evaluating dense checkpoint ...", flush=True)
    loss, perplexity = eval_perplexity(model, reader, args.device, args.eval_batches)
    payload = {
        "checkpoint": str(args.ckpt_path),
        "model_size": getattr(cfg, "model_size", None),
        "sequence_length": cfg.sequence_length,
        "eval_batches": args.eval_batches,
        "eval_batch_size": args.eval_batch_size,
        "val_loss": loss,
        "ppl": perplexity,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_path.with_suffix(args.output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output_path)
    print(f"VAL_LOSS={loss:.8f}", flush=True)
    print(f"PPL={perplexity:.8f}", flush=True)
    print(f"RESULT={args.output_path}", flush=True)


if __name__ == "__main__":
    main()
