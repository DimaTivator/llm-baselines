#!/usr/bin/env python3
"""Download the private 257M checkpoint bundle to persistent Cloud storage."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()

    args.destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.destination,
    )
    checkpoints = sorted(
        args.destination.glob("llama257m_*/ckpts/latest/main.pt")
    )
    print(f"STAGED_CHECKPOINTS={len(checkpoints)}", flush=True)
    if len(checkpoints) != 17:
        raise RuntimeError(f"Expected 17 checkpoints, found {len(checkpoints)}")
    calibration = args.destination / "calibration" / "val.bin"
    if not calibration.exists():
        raise FileNotFoundError(calibration)
    print(f"CALIBRATION_TOKENS={calibration}", flush=True)


if __name__ == "__main__":
    main()
