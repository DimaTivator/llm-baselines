#!/usr/bin/env python3
"""Run a resumable aligned SVD-LLM margin sweep over explicit checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


EXPERIMENT_RE = re.compile(
    r"(?:^|_)llama(?P<size>\d+)m_.*?_lr(?P<lr>[^_]+)_sl1_(?P<coef>[^_]+)_"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--auto_rank_multiple", type=int, default=16)
    parser.add_argument("--margins", type=int, nargs="+", required=True)
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--calib_batches", type=int, default=16)
    return parser.parse_args()


def checkpoint_metadata(checkpoint: Path) -> dict[str, str]:
    experiment = checkpoint.parents[2].name
    match = EXPERIMENT_RE.search(experiment)
    if match is None:
        raise ValueError(f"Unrecognized experiment name: {experiment}")
    return {"experiment": experiment, **match.groupdict()}


def result_dir(output_root: Path, metadata: dict[str, str]) -> Path:
    return (
        output_root
        / f"{metadata['size']}m"
        / f"coef_{metadata['coef']}_lr_{metadata['lr']}"
    )


def is_complete(output_dir: Path, auto_rank_multiple: int, margins: list[int]) -> bool:
    result_path = output_dir / "svdllm_margin_table.json"
    if not (output_dir / "COMPLETE").exists() or not result_path.exists():
        return False
    try:
        payload = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    completed_margins = {row.get("margin") for row in payload.get("rows", [])}
    return (
        payload.get("auto_rank_multiple") == auto_rank_multiple
        and payload.get("margins") == margins
        and None in completed_margins
        and set(margins).issubset(completed_margins)
    )


def collect_rows(output_root: Path) -> list[dict]:
    rows: list[dict] = []
    for result_path in output_root.glob("*m/*/svdllm_margin_table.json"):
        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        metric_rows = payload.get("rows", [])
        if not metric_rows:
            continue
        base = next((row for row in metric_rows if row.get("margin") is None), None)
        if base is None:
            continue
        for compressed in metric_rows:
            margin = compressed.get("margin")
            if margin is None:
                continue
            base_loss = float(base["val_loss"])
            compressed_loss = float(compressed["val_loss"])
            rows.append(
                {
                    "model_size": payload["model_size"],
                    "coefficient": float(payload["weight_decay"]),
                    "learning_rate": float(payload["learning_rate"]),
                    "margin": int(margin),
                    "compression_rate": float(compressed["comp_rate"]),
                    "mean_rank": float(compressed["mean_rank"]),
                    "min_rank": int(compressed["min_rank"]),
                    "base_loss": base_loss,
                    "compressed_loss": compressed_loss,
                    "delta_loss": compressed_loss - base_loss,
                    "relative_delta_loss_percent": (
                        100.0 * (compressed_loss - base_loss) / base_loss
                    ),
                    "base_ppl": float(base["ppl"]),
                    "compressed_ppl": float(compressed["ppl"]),
                    "delta_ppl": float(compressed["ppl"]) - float(base["ppl"]),
                    "checkpoint": payload["checkpoint"],
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            int(str(row["model_size"]).removesuffix("M")),
            row["coefficient"],
            row["learning_rate"],
            row["margin"],
        ),
    )


def build_markdown(
    rows: list[dict], total: int, auto_rank_multiple: int, margins: list[int]
) -> str:
    lines = [
        "# SVD-LLM aligned auto-rank margin quality sweep",
        "",
        (
            f"Completed checkpoints: "
            f"{len({row['checkpoint'] for row in rows})}/{total}. "
            f"`auto_rank_multiple={auto_rank_multiple}`, margins={margins}."
        ),
        "",
    ]
    sizes = sorted({row["model_size"] for row in rows}, key=lambda value: int(value[:-1]))
    for size in sizes:
        scale_rows = [row for row in rows if row["model_size"] == size]
        lines.extend(
            [
                f"## {size}",
                "",
                "| coef | lr | margin | CR | mean r | min r | base loss | compressed loss | Δloss | Δloss, % | base PPL | compressed PPL | ΔPPL |",
                "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in scale_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{row['coefficient']:g}",
                        f"{row['learning_rate']:g}",
                        f"{row['margin']:+d}",
                        f"{row['compression_rate']:.3f}×",
                        f"{row['mean_rank']:.1f}",
                        str(row["min_rank"]),
                        f"{row['base_loss']:.4f}",
                        f"{row['compressed_loss']:.4f}",
                        f"{row['delta_loss']:+.4f}",
                        f"{row['relative_delta_loss_percent']:+.2f}%",
                        f"{row['base_ppl']:.2f}",
                        f"{row['compressed_ppl']:.2f}",
                        f"{row['delta_ppl']:+.2f}",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def persist_aggregate(
    output_root: Path, total: int, auto_rank_multiple: int, margins: list[int]
) -> None:
    rows = collect_rows(output_root)
    payload = {
        "auto_rank_multiple": auto_rank_multiple,
        "margins": margins,
        "completed_checkpoints": len({row["checkpoint"] for row in rows}),
        "total_checkpoints": total,
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "aggregate.json"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(payload, indent=2) + "\n")
    json_tmp.replace(json_path)

    markdown_path = output_root / "aggregate.md"
    markdown_tmp = markdown_path.with_suffix(".md.tmp")
    markdown_tmp.write_text(build_markdown(rows, total, auto_rank_multiple, margins))
    markdown_tmp.replace(markdown_path)


def main() -> None:
    args = parse_args()
    if args.auto_rank_multiple < 16 or args.auto_rank_multiple % 16 != 0:
        raise ValueError("auto_rank_multiple must be a positive multiple of 16")
    if not args.margins or len(args.margins) != len(set(args.margins)):
        raise ValueError("margins must be a non-empty list of unique integers")

    checkpoints = []
    seen = set()
    for checkpoint in args.checkpoints:
        checkpoint = checkpoint.resolve()
        if checkpoint in seen:
            raise ValueError(f"Duplicate checkpoint: {checkpoint}")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        seen.add(checkpoint)
        checkpoints.append((checkpoint, checkpoint_metadata(checkpoint)))
    checkpoints.sort(
        key=lambda item: (
            int(item[1]["size"]), float(item[1]["coef"]), item[1]["lr"]
        )
    )

    persist_aggregate(
        args.output_root, len(checkpoints), args.auto_rank_multiple, args.margins
    )
    for index, (checkpoint, metadata) in enumerate(checkpoints, start=1):
        output_dir = result_dir(args.output_root, metadata)
        label = f"{metadata['size']}M coef={metadata['coef']} lr={metadata['lr']}"
        if is_complete(output_dir, args.auto_rank_multiple, args.margins):
            print(f"[{index}/{len(checkpoints)}] SKIP complete: {label}", flush=True)
            continue

        print(f"[{index}/{len(checkpoints)}] RUN {label}", flush=True)
        (output_dir / "COMPLETE").unlink(missing_ok=True)
        command = [
            sys.executable,
            "-m",
            "compression.svdllm_margin_table",
            "--ckpt_path",
            str(checkpoint),
            "--output_dir",
            str(output_dir),
            "--device",
            args.device,
            "--eval_batches",
            str(args.eval_batches),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--calib_batches",
            str(args.calib_batches),
            "--margins",
            *(str(margin) for margin in args.margins),
            "--auto_rank_multiple",
            str(args.auto_rank_multiple),
            "--no_downstream",
            "--model_size",
            f"{metadata['size']}M",
            "--weight_decay",
            metadata["coef"],
            "--learning_rate",
            metadata["lr"],
        ]
        subprocess.run(command, check=True)
        persist_aggregate(
            args.output_root, len(checkpoints), args.auto_rank_multiple, args.margins
        )

    persist_aggregate(
        args.output_root, len(checkpoints), args.auto_rank_multiple, args.margins
    )
    print(f"Sweep complete: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
