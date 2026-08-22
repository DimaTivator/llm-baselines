#!/usr/bin/env python3
"""Upload the three L2-WD speed checkpoints as a private Cloud staging bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def checkpoint_argument(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=/absolute/path/main.pt")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file() or path.name != "main.pt":
        raise argparse.ArgumentTypeError(f"missing main.pt: {path}")
    return label, path


def summary_for(checkpoint: Path) -> Path:
    for parent in checkpoint.parents:
        summary = parent / "summary.json"
        if summary.is_file():
            return summary
    raise FileNotFoundError(f"Could not locate summary.json above {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--checkpoint", action="append", type=checkpoint_argument, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()

    if len(args.checkpoint) != 3:
        raise ValueError("Expected exactly three checkpoints")
    labels = [label for label, _ in args.checkpoint]
    if len(labels) != len(set(labels)):
        raise ValueError("Checkpoint labels must be unique")
    calibration = args.calibration.resolve()
    if not calibration.is_file():
        raise FileNotFoundError(calibration)

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    for label, checkpoint in args.checkpoint:
        summary = summary_for(checkpoint)
        for local_path, remote_path in (
            (checkpoint, f"{label}/ckpts/latest/main.pt"),
            (summary, f"{label}/summary.json"),
        ):
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=args.repo_id,
                repo_type="dataset",
            )
            print(f"UPLOADED={remote_path} bytes={local_path.stat().st_size}", flush=True)
    api.upload_file(
        path_or_fileobj=calibration,
        path_in_repo="calibration/val.bin",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"UPLOADED=calibration/val.bin bytes={calibration.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
