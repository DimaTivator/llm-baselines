#!/usr/bin/env python3
"""Upload selected model-only checkpoints and calibration tokens to HF."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi


def _experiment_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No experiments listed in {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate experiments in {path}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True, type=Path)
    parser.add_argument("--experiment_list", required=True, type=Path)
    parser.add_argument("--calibration_tokens", required=True, type=Path)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--temp_dir", required=True, type=Path)
    parser.add_argument(
        "--manifest_path",
        default="manifests/svdllm-m16-speed-selected.json",
    )
    args = parser.parse_args()

    experiments = _experiment_names(args.experiment_list)
    if not args.calibration_tokens.is_file():
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
    for index, experiment_name in enumerate(experiments, start=1):
        experiment = args.source_root / experiment_name
        checkpoint = experiment / "ckpts" / "latest" / "main.pt"
        summary = experiment / "summary.json"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not summary.is_file():
            raise FileNotFoundError(summary)

        checkpoint_target = f"{experiment_name}/ckpts/latest/main.pt"
        summary_target = f"{experiment_name}/summary.json"
        print(f"[{index}/{len(experiments)}] {experiment_name}", flush=True)

        stripped_bytes: int | None = None
        if checkpoint_target not in uploaded:
            with tempfile.TemporaryDirectory(dir=args.temp_dir) as temporary:
                stripped_path = Path(temporary) / "main.pt"
                artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if "model" not in artifact:
                    raise KeyError(f"Missing model state in {checkpoint}")
                torch.save({"model": artifact["model"]}, stripped_path)
                del artifact
                stripped_bytes = stripped_path.stat().st_size
                print(
                    f"  uploading model-only checkpoint ({stripped_bytes / 2**30:.2f} GiB)",
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
                path_or_fileobj=summary,
                path_in_repo=summary_target,
                repo_id=args.repo_id,
                repo_type="dataset",
            )
            uploaded.add(summary_target)

        manifest.append(
            {
                "experiment": experiment_name,
                "source_checkpoint": str(checkpoint),
                "source_bytes": checkpoint.stat().st_size,
                "model_only_bytes": stripped_bytes or 0,
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
            path_in_repo=args.manifest_path,
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print(f"UPLOADED_SELECTION={len(experiments)}", flush=True)
    print(f"HF_REPO={args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
