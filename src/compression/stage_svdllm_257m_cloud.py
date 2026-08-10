#!/usr/bin/env python3
"""Download the private 257M checkpoint bundle to persistent Cloud storage."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--expected_checkpoints", default=17, type=int)
    parser.add_argument(
        "--allow_patterns",
        nargs="+",
        default=None,
        help="Optional Hugging Face snapshot patterns for a partial download.",
    )
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
            allow_patterns=args.allow_patterns,
        )
    checkpoints = sorted(
        args.destination.glob("llama257m_*/ckpts/latest/main.pt")
    )
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoints)
    print(f"STAGED_CHECKPOINTS={len(checkpoints)}", flush=True)
    usage = shutil.disk_usage(args.destination)
    print(f"STAGED_CHECKPOINT_BYTES={checkpoint_bytes}", flush=True)
    print(f"FILESYSTEM_FREE_BYTES={usage.free}", flush=True)
    if args.inspect_only:
        return
    if len(checkpoints) != args.expected_checkpoints:
        raise RuntimeError(
            f"Expected {args.expected_checkpoints} checkpoints, found {len(checkpoints)}"
        )
    calibration = args.destination / "calibration" / "val.bin"
    if not calibration.exists():
        raise FileNotFoundError(calibration)
    print(f"CALIBRATION_TOKENS={calibration}", flush=True)


if __name__ == "__main__":
    main()
