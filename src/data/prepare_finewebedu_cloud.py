import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SHARDS = tuple(f"000_{index:05d}.parquet" for index in range(8))
MAX_SHARD_BYTES = 2_300_000_000
FREE_SPACE_MARGIN_BYTES = 5_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=len(SHARDS))
    args = parser.parse_args()

    if not 1 <= args.num_shards <= len(SHARDS):
        raise ValueError(f"--num-shards must be in [1, {len(SHARDS)}]")

    dataset_dir = args.root / "sample" / "100BT"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    requested = SHARDS[: args.num_shards]
    missing = [name for name in requested if not (dataset_dir / name).is_file()]
    required = len(missing) * MAX_SHARD_BYTES + FREE_SPACE_MARGIN_BYTES
    free = shutil.disk_usage(dataset_dir).free
    if free < required:
        raise RuntimeError(
            f"Insufficient free space for {len(missing)} FineWeb shards: "
            f"need at least {required / 1e9:.1f} GB, have {free / 1e9:.1f} GB"
        )

    for name in missing:
        print(f"Downloading FineWeb-Edu shard {name}", flush=True)
        hf_hub_download(
            repo_id="HuggingFaceFW/fineweb-edu",
            repo_type="dataset",
            revision=REVISION,
            filename=f"sample/100BT/{name}",
            local_dir=args.root,
        )

    paths = [dataset_dir / name for name in requested]
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        raise RuntimeError("FineWeb-Edu shard preparation is incomplete")

    total = sum(path.stat().st_size for path in paths)
    print(
        f"FineWeb-Edu ready: {len(paths)} shards, {total / 1e9:.2f} GB at {dataset_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
