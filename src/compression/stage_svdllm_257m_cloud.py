#!/usr/bin/env python3
"""Download the private 257M checkpoint bundle to persistent Cloud storage."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Report already staged files without downloading anything.",
    )
    args = parser.parse_args()

    args.destination.mkdir(parents=True, exist_ok=True)
    if not args.inspect_only:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.destination,
            cache_dir=args.destination.parent / ".hf-cache",
        )
    checkpoints = sorted(
        args.destination.glob("llama257m_*/ckpts/latest/main.pt")
    )
    print(f"STAGED_CHECKPOINTS={len(checkpoints)}", flush=True)
    usage = shutil.disk_usage(args.destination)
    print(f"DESTINATION_BYTES={sum(path.stat().st_size for path in args.destination.rglob('*') if path.is_file())}", flush=True)
    print(f"FILESYSTEM_FREE_BYTES={usage.free}", flush=True)
    if args.inspect_only:
        return
    if len(checkpoints) != 17:
        raise RuntimeError(f"Expected 17 checkpoints, found {len(checkpoints)}")
    calibration = args.destination / "calibration" / "val.bin"
    if not calibration.exists():
        raise FileNotFoundError(calibration)
    print(f"CALIBRATION_TOKENS={calibration}", flush=True)


if __name__ == "__main__":
    main()
