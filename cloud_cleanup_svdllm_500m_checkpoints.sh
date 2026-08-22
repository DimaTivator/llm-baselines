#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace-SR006.nfs2/dimativator/spectral-wd-500m}"
RESULT_DIR="${RESULT_DIR:-/workspace-SR006.nfs2/dimativator/results/svdllm-m16-compile-b256-500m}"
EXPECTED_ROOT="${EXPECTED_ROOT:-/workspace-SR006.nfs2/dimativator/spectral-wd-500m}"
EXPECTED_RESULT="${EXPECTED_RESULT:-/workspace-SR006.nfs2/dimativator/results/svdllm-m16-compile-b256-500m}"
EXPECTED_CHECKPOINTS="${EXPECTED_CHECKPOINTS:-6}"
HF_CACHE_DIR="/workspace-SR006.nfs2/dimativator/.hf-cache/datasets--DimaTivator--spectral-wd-500m-checkpoints"
OUTPUT_PATH="${RESULT_DIR}/results.json"

if [[ "$(realpath -m "${CHECKPOINT_ROOT}")" != "${EXPECTED_ROOT}" ]]; then
    echo "Refusing unexpected CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
    exit 1
fi
if [[ "$(realpath -m "${RESULT_DIR}")" != "${EXPECTED_RESULT}" ]]; then
    echo "Refusing unexpected RESULT_DIR=${RESULT_DIR}"
    exit 1
fi
if [[ ! -f "${OUTPUT_PATH}" ]]; then
    echo "Missing benchmark result: ${OUTPUT_PATH}"
    exit 1
fi

python - "${OUTPUT_PATH}" "${EXPECTED_CHECKPOINTS}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
if len(payload.get("checkpoints", [])) != expected:
    raise SystemExit(f"Refusing cleanup: expected {expected} completed checkpoints")
print(f"RESULT_VALIDATED_CHECKPOINTS={expected}")
PY

if [[ -d "${CHECKPOINT_ROOT}" ]]; then
    rm -rf "${CHECKPOINT_ROOT}"
fi
if [[ -d "${HF_CACHE_DIR}" ]]; then
    rm -rf "${HF_CACHE_DIR}"
fi

if [[ -e "${CHECKPOINT_ROOT}" || -e "${HF_CACHE_DIR}" ]]; then
    echo "Cleanup verification failed"
    exit 1
fi
echo "CLOUD_500M_CHECKPOINT_CLEANUP=OK"
echo "RESULT_PRESERVED=${OUTPUT_PATH}"
