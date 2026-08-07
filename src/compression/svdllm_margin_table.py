#!/usr/bin/env python3
"""Evaluate one checkpoint with SVD-LLM auto-rank plus rank margins.

The output is written after the baseline and after every margin so an interrupted
Cloud.ru job still leaves all completed rows on persistent storage.
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
from compression.benchmark import (
    _DOWNSTREAM_TASKS,
    _TASK_SHORT,
    _compression_rate,
    _run_downstream,
)
from compression.svd_llm import apply_svd_llm, collect_input_covariance
from data.utils import DataReader, get_dataset


DEFAULT_MARGINS = (-8, -4, 0, 4, 8, 12, 16, 20)
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


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.{digits}f}"


def build_markdown(
    rows: list[dict],
    tasks: list[str],
    global_base_loss: float | None,
) -> str:
    header = [
        "m",
        "CR",
        "mean r",
        "min r",
        "n",
        "val loss",
        "Δown",
        "Δglobal",
        "PPL",
        *(_TASK_SHORT[task] for task in tasks),
    ]
    display_names = {
        "arc_e": "ARC-E",
        "arc_c": "ARC-C",
        "gsm8k": "GSM8K bpb↓",
        "hella": "Hella",
        "piqa": "PIQA",
    }
    header = [display_names.get(cell, cell) for cell in header]
    lines = [
        "| " + " | ".join(header) + " |",
        "|:--|" + "|".join("---:" for _ in header[1:]) + "|",
    ]

    own_base_loss = rows[0]["val_loss"]
    for row in rows:
        margin = row["margin"]
        cells = [
            "base" if margin is None else f"{margin:+d}",
            f"{row['comp_rate']:.3f}×",
            "—" if row["mean_rank"] is None else f"{row['mean_rank']:.1f}",
            "—" if row["min_rank"] is None else str(row["min_rank"]),
            str(row["n_compressed"]),
            f"{row['val_loss']:.4f}",
            f"{row['val_loss'] - own_base_loss:+.4f}",
            (
                "N/A"
                if global_base_loss is None
                else f"{row['val_loss'] - global_base_loss:+.4f}"
            ),
            f"{row['ppl']:.2f}",
        ]
        cells.extend(format_float(row.get(task)) for task in tasks)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def persist_results(
    output_dir: Path,
    metadata: dict,
    rows: list[dict],
    tasks: list[str],
    global_base_loss: float | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **metadata,
        "global_base_loss": global_base_loss,
        "tasks": tasks,
        "rows": rows,
    }
    with (output_dir / "svdllm_margin_table.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    (output_dir / "svdllm_margin_table.md").write_text(
        build_markdown(rows, tasks, global_base_loss)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--calib_batches", type=int, default=16)
    parser.add_argument("--margins", type=int, nargs="+", default=DEFAULT_MARGINS)
    parser.add_argument("--global_base_loss", type=float, default=None)
    parser.add_argument("--no_downstream", action="store_true")
    parser.add_argument("--model_size", required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, cfg, _ = load_model_from_ckpt(args.ckpt_path, device=args.device)
    cfg.eval_batch_size = args.eval_batch_size
    original = copy.deepcopy(model)

    calibration_reader = build_val_reader(cfg, args.eval_batch_size)
    calibration_data = make_calibration_dataloader(
        calibration_reader, args.calib_batches
    )
    print(
        f"Collecting SVD-LLM covariance on {args.calib_batches} batches ...",
        flush=True,
    )
    whitening_stats = collect_input_covariance(
        original,
        calibration_data,
        n_batches=len(calibration_data),
        device=args.device,
    )

    val_reader = build_val_reader(cfg, args.eval_batch_size)
    print("Evaluating uncompressed baseline ...", flush=True)
    base_loss, base_ppl = eval_perplexity(
        original, val_reader, args.device, args.eval_batches
    )
    base_downstream = (
        {} if args.no_downstream else _run_downstream(original, cfg, args.device)
    )
    tasks = [task for task in _DOWNSTREAM_TASKS if task in base_downstream]
    rows = [
        {
            "margin": None,
            "comp_rate": 1.0,
            "mean_rank": None,
            "min_rank": None,
            "n_compressed": 0,
            "val_loss": base_loss,
            "ppl": base_ppl,
            **{task: base_downstream.get(task) for task in tasks},
        }
    ]
    metadata = {
        "checkpoint": str(args.ckpt_path),
        "experiment_name": getattr(cfg, "experiment_name", None),
        "model_size": args.model_size,
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate,
        "method": "svd_llm",
        "rank_rule": "round(effective_rank(W)) + margin",
        "target_modules": list(TARGET_MODULES),
        "margins": args.margins,
        "eval_batches": args.eval_batches,
        "eval_batch_size": args.eval_batch_size,
        "calib_batches": args.calib_batches,
    }
    persist_results(
        args.output_dir, metadata, rows, tasks, args.global_base_loss
    )

    for margin in args.margins:
        compressed = copy.deepcopy(original)
        _, comp_info = apply_svd_llm(
            compressed,
            rank="auto",
            whitening_stats=whitening_stats,
            target_modules=TARGET_MODULES,
            device=args.device,
            margin=margin,
        )
        ranks = list(comp_info.values())
        val_reader.set_step(0)
        val_loss, ppl = eval_perplexity(
            compressed, val_reader, args.device, args.eval_batches
        )
        downstream = (
            {} if args.no_downstream else _run_downstream(compressed, cfg, args.device)
        )
        row = {
            "margin": margin,
            "comp_rate": _compression_rate(original, comp_info),
            "mean_rank": sum(ranks) / len(ranks) if ranks else None,
            "min_rank": min(ranks) if ranks else None,
            "n_compressed": len(ranks),
            "val_loss": val_loss,
            "ppl": ppl,
            **{task: downstream.get(task) for task in tasks},
        }
        rows.append(row)
        persist_results(
            args.output_dir, metadata, rows, tasks, args.global_base_loss
        )
        print(
            f"margin={margin:+d} CR={row['comp_rate']:.3f}x "
            f"mean_rank={row['mean_rank']:.1f} val_loss={val_loss:.4f}",
            flush=True,
        )
        del compressed
        torch.cuda.empty_cache()

    (args.output_dir / "COMPLETE").write_text("complete\n")
    print(f"Table complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
