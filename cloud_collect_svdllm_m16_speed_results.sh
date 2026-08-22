#!/usr/bin/env bash

set -euo pipefail

RESULT_DIR=${RESULT_DIR:?RESULT_DIR is required}
EXPECTED_CHECKPOINTS=${EXPECTED_CHECKPOINTS:?EXPECTED_CHECKPOINTS is required}
HF_REPO_ID=${HF_REPO_ID:-DimaTivator/spectral-wd-500m-checkpoints}
RESULT_UPLOAD_PREFIX=${RESULT_UPLOAD_PREFIX:?RESULT_UPLOAD_PREFIX is required}
OUTPUT_PATH="${RESULT_DIR}/results.json"
GPU_PROCESS_LOG="${RESULT_DIR}/gpu-processes.csv"

python - "${OUTPUT_PATH}" "${GPU_PROCESS_LOG}" "${EXPECTED_CHECKPOINTS}" <<'PY'
import csv
import json
import sys

output_path, process_path, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
payload = json.load(open(output_path))
checkpoints = payload.get("checkpoints", [])
if len(checkpoints) != expected:
    raise SystemExit(f"Expected {expected} checkpoints, found {len(checkpoints)}")

pids = set()
with open(process_path) as handle:
    for row in csv.reader(handle):
        if len(row) >= 2 and row[1].strip().isdigit():
            pids.add(row[1].strip())
if len(pids) != 1:
    raise SystemExit(f"Expected one benchmark PID, found {sorted(pids)}")

print(f"RESULT_VALIDATED_CHECKPOINTS={len(checkpoints)}")
print(f"WATCHDOG_PIDS={','.join(sorted(pids))}")
for entry in checkpoints:
    comparison = next(
        row for row in entry["comparisons"] if row["batch_size"] == 256
    )
    measurements = entry["measurements"]
    dense = next(
        row for row in measurements
        if row["batch_size"] == 256 and row["model"] == "original"
    )
    compressed = next(
        row for row in measurements
        if row["batch_size"] == 256 and row["model"] == "compressed"
    )
    print(
        "CHECKPOINT=" + entry["checkpoint"]
        + f" DENSE_MS={dense['latency_ms']:.4f}"
        + f" COMPRESSED_MS={compressed['latency_ms']:.4f}"
        + f" SPEEDUP={comparison['speedup']:.4f}",
        flush=True,
    )
PY

python - "${OUTPUT_PATH}" "${GPU_PROCESS_LOG}" "${HF_REPO_ID}" "${RESULT_UPLOAD_PREFIX}" <<'PY'
import sys

from huggingface_hub import HfApi

output_path, process_path, repo_id, prefix = sys.argv[1:]
api = HfApi()
for local_path, name in ((output_path, "results.json"), (process_path, "gpu-processes.csv")):
    remote_path = f"{prefix.rstrip('/')}/{name}"
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"RESULT_UPLOADED={remote_path}", flush=True)
PY
