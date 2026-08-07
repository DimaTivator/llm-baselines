#!/usr/bin/env python3
"""Upload stripped 257M checkpoints and calibration tokens to a private HF dataset."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True, type=Path)
    parser.add_argument("--calibration_tokens", required=True, type=Path)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--temp_dir", required=True, type=Path)
    args = parser.parse_args()

    checkpoints = sorted(args.source_root.glob("llama257m_*/ckpts/latest/main.pt"))
    if len(checkpoints) != 17:
        raise RuntimeError(f"Expected 17 checkpoints, found {len(checkpoints)}")
    if not args.calibration_tokens.exists():
        raise FileNotFoundError(args.calibration_tokens)

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    uploaded = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
    manifest: list[dict[str, str | int]] = []

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    for index, checkpoint in enumerate(checkpoints, start=1):
        experiment = checkpoint.parents[2]
        checkpoint_target = f"{experiment.name}/ckpts/latest/main.pt"
        summary_target = f"{experiment.name}/summary.json"
        print(f"[{index}/{len(checkpoints)}] {experiment.name}", flush=True)

        if checkpoint_target not in uploaded:
            with tempfile.TemporaryDirectory(dir=args.temp_dir) as temporary:
                stripped_path = Path(temporary) / "main.pt"
                artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
                torch.save({"model": artifact["model"]}, stripped_path)
                del artifact
                print(
                    f"  uploading stripped checkpoint ({stripped_path.stat().st_size / 2**30:.2f} GiB)",
                    flush=True,
                )
                api.upload_file(
                    path_or_fileobj=stripped_path,
                    path_in_repo=checkpoint_target,
                    repo_id=args.repo_id,
                    repo_type="dataset",
                )
            uploaded.add(checkpoint_target)
        else:
            print("  checkpoint already uploaded", flush=True)

        if summary_target not in uploaded:
            api.upload_file(
                path_or_fileobj=experiment / "summary.json",
                path_in_repo=summary_target,
                repo_id=args.repo_id,
                repo_type="dataset",
            )
            uploaded.add(summary_target)

        manifest.append(
            {
                "experiment": experiment.name,
                "source_checkpoint": str(checkpoint),
                "source_bytes": checkpoint.stat().st_size,
                "checkpoint_path": checkpoint_target,
                "summary_path": summary_target,
            }
        )

    calibration_target = "calibration/val.bin"
    if calibration_target not in uploaded:
        api.upload_file(
            path_or_fileobj=args.calibration_tokens,
            path_in_repo=calibration_target,
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    with tempfile.TemporaryDirectory(dir=args.temp_dir) as temporary:
        manifest_path = Path(temporary) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        api.upload_file(
            path_or_fileobj=manifest_path,
            path_in_repo="manifest.json",
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print(f"UPLOADED_CHECKPOINTS={len(checkpoints)}", flush=True)
    print(f"HF_REPO={args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
