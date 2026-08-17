#!/usr/bin/env bash

set -u

RESULT_DIR=${RESULT_DIR:-/workspace-SR006.nfs2/dimativator/results/svdllm-m16-compile-b256-selected}
OUTPUT_PATH="${RESULT_DIR}/results.json"
GPU_PROCESS_LOG="${RESULT_DIR}/gpu-processes.csv"

if [[ -f "${OUTPUT_PATH}" ]]; then
    python - "${OUTPUT_PATH}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
print(f"RESULT_PRESENT=1")
print(f"COMPLETED_CHECKPOINTS={len(payload.get('checkpoints', []))}")
if payload.get("checkpoints"):
    print(f"LAST_CHECKPOINT={payload['checkpoints'][-1]['checkpoint']}")
PY
else
    echo "RESULT_PRESENT=0"
    echo "COMPLETED_CHECKPOINTS=0"
fi

if [[ -f "${GPU_PROCESS_LOG}" ]]; then
    echo "GPU_PROCESS_LOG_PRESENT=1"
    echo "GPU_PROCESS_LOG_LINES=$(wc -l < "${GPU_PROCESS_LOG}")"
    tail -10 "${GPU_PROCESS_LOG}"
else
    echo "GPU_PROCESS_LOG_PRESENT=0"
fi
