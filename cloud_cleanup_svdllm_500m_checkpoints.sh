#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace-SR006.nfs2/dimativator/spectral-wd-500m}"
RESULT_DIR="${RESULT_DIR:-/workspace-SR006.nfs2/dimativator/results/svdllm-m16-compile-b256-500m}"
EXPECTED_ROOT="${EXPECTED_ROOT:-/workspace-SR006.nfs2/dimativator/spectral-wd-500m}"
EXPECTED_RESULT="${EXPECTED_RESULT:-/workspace-SR006.nfs2/dimativator/results/svdllm-m16-compile-b256-500m}"
EXPECTED_CHECKPOINTS="${EXPECTED_CHECKPOINTS:-6}"
EXPECTED_ROWS="${EXPECTED_ROWS:-${EXPECTED_CHECKPOINTS}}"
EXPECTED_MARGINS="${EXPECTED_MARGINS:-}"
EXPECTED_MARGIN_PRESET="${EXPECTED_MARGIN_PRESET:-}"
EXPECTED_AUTO_RANK_MULTIPLE="${EXPECTED_AUTO_RANK_MULTIPLE:-}"
EXPECTED_COMPILE_MODE="${EXPECTED_COMPILE_MODE:-}"
EXPECTED_RESIDUAL_GUARD="${EXPECTED_RESIDUAL_GUARD:-}"
HF_CACHE_DIR="/workspace-SR006.nfs2/dimativator/.hf-cache/datasets--DimaTivator--spectral-wd-500m-checkpoints"
OUTPUT_PATH="${RESULT_DIR}/results.json"

if [[ -n "${EXPECTED_MARGIN_PRESET}" ]]; then
    if [[ "${EXPECTED_MARGIN_PRESET}" != "symmetric30" ]]; then
        echo "Unknown EXPECTED_MARGIN_PRESET=${EXPECTED_MARGIN_PRESET}"
        exit 1
    fi
    if [[ -n "${EXPECTED_MARGINS}" ]]; then
        echo "Set only one of EXPECTED_MARGIN_PRESET or EXPECTED_MARGINS"
        exit 1
    fi
    EXPECTED_MARGINS="0 -10 10 -15 15 -20 20 -25 25 -30 30"
fi

if [[ ! "${EXPECTED_ROOT}" =~ ^/workspace-SR006\.nfs2/dimativator/spectral-wd-500m(-[a-zA-Z0-9._-]+)?$ ]]; then
    echo "Refusing unsafe EXPECTED_ROOT=${EXPECTED_ROOT}"
    exit 1
fi
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
if [[ ! -f "${RESULT_DIR}/BENCHMARK_EXIT_0" ]]; then
    echo "Benchmark did not report exit 0"
    exit 1
fi

python - "${OUTPUT_PATH}" "${EXPECTED_CHECKPOINTS}" "${EXPECTED_ROWS}" "${EXPECTED_MARGINS}" "${EXPECTED_AUTO_RANK_MULTIPLE}" "${EXPECTED_COMPILE_MODE}" "${EXPECTED_RESIDUAL_GUARD}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

payload = json.load(open(sys.argv[1]))
expected_checkpoints = int(sys.argv[2])
expected_rows = int(sys.argv[3])
expected_margins = {int(value) for value in sys.argv[4].replace(",", " ").split()} if sys.argv[4] else None
expected_multiple = int(sys.argv[5]) if sys.argv[5] else None
expected_compile_mode = sys.argv[6] or None
expected_guard = float(sys.argv[7]) if sys.argv[7] else None
entries = payload.get("checkpoints", [])
if len(entries) != expected_rows:
    raise SystemExit(f"Refusing cleanup: expected {expected_rows} rows, found {len(entries)}")
if expected_multiple is not None and payload.get("auto_rank_multiple") != expected_multiple:
    raise SystemExit("Refusing cleanup: wrong auto_rank_multiple")
if expected_compile_mode is not None and payload.get("compile_mode") != expected_compile_mode:
    raise SystemExit("Refusing cleanup: wrong compile mode")
if expected_guard is not None and payload.get("max_whitened_relative_residual") != expected_guard:
    raise SystemExit("Refusing cleanup: wrong residual guard")
by_label = defaultdict(set)
for row in entries:
    label = Path(row["checkpoint"]).parents[2].name
    by_label[label].add(int(row.get("margin", 0)))
    if row.get("factor_order_check", {}).get("factor_order") != "B_then_A":
        raise SystemExit(f"Refusing cleanup: wrong factor order for {label}")
if len(by_label) != expected_checkpoints:
    raise SystemExit(f"Refusing cleanup: expected {expected_checkpoints} checkpoint labels")
if expected_margins is not None:
    for label, margins in by_label.items():
        if margins != expected_margins:
            raise SystemExit(f"Refusing cleanup: wrong margins for {label}")
print(f"RESULT_VALIDATED_CHECKPOINTS={expected_checkpoints}")
print(f"RESULT_VALIDATED_ROWS={expected_rows}")
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
