#!/usr/bin/env python3
"""Stage an explicit checkpoint selection on persistent Cloud storage."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def _experiment_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No experiments listed in {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate experiments in {path}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--experiment_list", required=True, type=Path)
    parser.add_argument("--inspect_only", action="store_true")
    args = parser.parse_args()

    experiments = _experiment_names(args.experiment_list)
    allow_patterns = ["calibration/val.bin"]
    for experiment in experiments:
        allow_patterns.extend(
            (
                f"{experiment}/ckpts/latest/main.pt",
                f"{experiment}/summary.json",
            )
        )

    args.destination.mkdir(parents=True, exist_ok=True)
    if not args.inspect_only:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.destination,
            cache_dir=args.destination.parent / ".hf-cache",
            allow_patterns=allow_patterns,
        )

    checkpoints = [
        args.destination / experiment / "ckpts" / "latest" / "main.pt"
        for experiment in experiments
    ]
    missing = [path for path in checkpoints if not path.is_file()]
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoints if path.is_file())
    calibration = args.destination / "calibration" / "val.bin"
    usage = shutil.disk_usage(args.destination)
    print(f"SELECTED_EXPERIMENTS={len(experiments)}", flush=True)
    print(f"STAGED_CHECKPOINTS={len(checkpoints) - len(missing)}", flush=True)
    print(f"STAGED_CHECKPOINT_BYTES={checkpoint_bytes}", flush=True)
    print(f"FILESYSTEM_FREE_BYTES={usage.free}", flush=True)
    print(f"CALIBRATION_PRESENT={int(calibration.is_file())}", flush=True)
    if missing:
        for path in missing:
            print(f"MISSING_CHECKPOINT={path}", flush=True)
    if args.inspect_only:
        return
    if missing:
        raise RuntimeError(f"Missing {len(missing)} selected checkpoints")
    if not calibration.is_file():
        raise FileNotFoundError(calibration)


if __name__ == "__main__":
    main()
