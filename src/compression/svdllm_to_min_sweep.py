#!/usr/bin/env python3
"""Evaluate an SVD-LLM auto-rank margin sweep down to rank one.

This is intentionally separate from the tensor-core-aligned ``m16`` protocol:
it retains unrounded ranks and disables only the optional residual rank floor,
so that a negative-margin sweep can reach the actual minimum rank.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from compression._eval_utils import (
    eval_perplexity,
    load_model_from_ckpt,
    make_calibration_dataloader,
)
from compression.benchmark import _compression_rate
from compression.svd_llm import apply_svd_llm, collect_input_covariance
from data.utils import DataReader, get_dataset


TARGET_MODULES = ("c_attn", "c_proj", "w1", "w2")


def build_val_reader(cfg: SimpleNamespace, batch_size: int) -> DataReader:
    return DataReader(
        data_src=get_dataset(cfg)["val"],
        batch_size=batch_size,
        sequence_length=cfg.sequence_length,
        seed=cfg.data_seed,
        with_replacement=False,
        auto_shard=False,
    )


def margins_to_min(min_margin: int) -> list[int]:
    if min_margin > -100 or min_margin % 50:
        raise ValueError("min_margin must be a multiple of 50 no greater than -100")
    return [0, -10, -25, -50, *range(-100, min_margin - 1, -50)]


def build_markdown(rows: list[dict]) -> str:
    header = [
        "margin",
        "CR",
        "mean rank",
        "min rank",
        "layers",
        "val loss",
        "Δloss",
        "Δloss, %",
        "PPL",
        "ΔPPL",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|:--|" + "|".join("---:" for _ in header[1:]) + "|",
    ]
    base = rows[0]
    for row in rows:
        delta_loss = row["val_loss"] - base["val_loss"]
        delta_ppl = row["ppl"] - base["ppl"]
        cells = [
            "dense" if row["margin"] is None else f"{row['margin']:+d}",
            f"{row['comp_rate']:.3f}×",
            "—" if row["mean_rank"] is None else f"{row['mean_rank']:.1f}",
            "—" if row["min_rank"] is None else str(row["min_rank"]),
            str(row["n_compressed"]),
            f"{row['val_loss']:.4f}",
            f"{delta_loss:+.4f}",
            f"{100 * delta_loss / base['val_loss']:+.2f}%",
            f"{row['ppl']:.2f}",
            f"{delta_ppl:+.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def persist(output_dir: Path, metadata: dict, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {**metadata, "rows": rows}
    json_path = output_dir / "results.json"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(payload, indent=2) + "\n")
    json_tmp.replace(json_path)
    markdown_path = output_dir / "results.md"
    markdown_tmp = markdown_path.with_suffix(".md.tmp")
    markdown_tmp.write_text(build_markdown(rows))
    markdown_tmp.replace(markdown_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--calib_batches", type=int, default=16)
    parser.add_argument("--min_margin", type=int, required=True)
    parser.add_argument("--model_size", required=True)
    parser.add_argument("--checkpoint_label", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    margins = margins_to_min(args.min_margin)
    model, cfg, _ = load_model_from_ckpt(args.ckpt_path, device=args.device)
    cfg.eval_batch_size = args.eval_batch_size
    original = model

    calibration_reader = build_val_reader(cfg, args.eval_batch_size)
    calibration_data = make_calibration_dataloader(
        calibration_reader, args.calib_batches
    )
    print(f"Collecting covariance on {args.calib_batches} calibration batches ...", flush=True)
    whitening_stats = collect_input_covariance(
        original,
        calibration_data,
        n_batches=len(calibration_data),
        device=args.device,
    )
    val_reader = build_val_reader(cfg, args.eval_batch_size)
    print("Evaluating dense baseline ...", flush=True)
    base_loss, base_ppl = eval_perplexity(
        original, val_reader, args.device, args.eval_batches
    )
    rows = [{
        "margin": None,
        "comp_rate": 1.0,
        "mean_rank": None,
        "min_rank": None,
        "n_compressed": 0,
        "val_loss": base_loss,
        "ppl": base_ppl,
    }]
    metadata = {
        "checkpoint": str(args.ckpt_path),
        "checkpoint_label": args.checkpoint_label,
        "experiment_name": getattr(cfg, "experiment_name", None),
        "model_size": args.model_size,
        "method": "svd_llm",
        "rank_rule": "max(1, round(effective_rank(W)) + margin)",
        "auto_rank_multiple": None,
        "max_whitened_relative_residual": None,
        "target_modules": list(TARGET_MODULES),
        "margins_requested": margins,
        "eval_batches": args.eval_batches,
        "eval_batch_size": args.eval_batch_size,
        "calib_batches": args.calib_batches,
    }
    persist(args.output_dir, metadata, rows)

    for margin in margins:
        compressed = copy.deepcopy(original)
        _, comp_info = apply_svd_llm(
            compressed,
            rank="auto",
            whitening_stats=whitening_stats,
            target_modules=TARGET_MODULES,
            device=args.device,
            margin=margin,
            auto_rank_multiple=None,
            max_whitened_relative_residual=None,
        )
        ranks = list(comp_info.values())
        val_reader.set_step(0)
        val_loss, ppl = eval_perplexity(
            compressed, val_reader, args.device, args.eval_batches
        )
        row = {
            "margin": margin,
            "comp_rate": _compression_rate(original, comp_info),
            "mean_rank": sum(ranks) / len(ranks) if ranks else None,
            "min_rank": min(ranks) if ranks else None,
            "n_compressed": len(ranks),
            "val_loss": val_loss,
            "ppl": ppl,
        }
        rows.append(row)
        persist(args.output_dir, metadata, rows)
        print(
            f"margin={margin:+d} CR={row['comp_rate']:.3f}x "
            f"mean_rank={row['mean_rank']:.1f} val_loss={val_loss:.4f}",
            flush=True,
        )
        all_rank_one = bool(ranks) and max(ranks) == 1
        del compressed
        torch.cuda.empty_cache()
        if all_rank_one:
            print("All compressed layers reached rank one; stopping sweep.", flush=True)
            break

    (args.output_dir / "COMPLETE").write_text("complete\n")
    print(f"Sweep complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
