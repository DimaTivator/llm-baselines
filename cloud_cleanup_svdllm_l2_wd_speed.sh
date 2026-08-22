#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace-SR006.nfs2/dimativator/svdllm-l2-wd-speed-20260822}"
RESULT_DIR="${RESULT_DIR:-/workspace-SR006.nfs2/dimativator/results/svdllm-l2-wd-speed-20260822}"
EXPECTED_ROOT="/workspace-SR006.nfs2/dimativator/svdllm-l2-wd-speed-20260822"
EXPECTED_RESULT="/workspace-SR006.nfs2/dimativator/results/svdllm-l2-wd-speed-20260822"
HF_CACHE_DIR="/workspace-SR006.nfs2/dimativator/.hf-cache/datasets--DimaTivator--svdllm-l2-wd-speed-checkpoints"
OUTPUT_PATH="${RESULT_DIR}/results.json"
GPU_PROCESS_LOG="${RESULT_DIR}/gpu-processes.csv"

[[ "$(realpath -m "${CHECKPOINT_ROOT}")" == "${EXPECTED_ROOT}" ]] || { echo "Refusing CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"; exit 1; }
[[ "$(realpath -m "${RESULT_DIR}")" == "${EXPECTED_RESULT}" ]] || { echo "Refusing RESULT_DIR=${RESULT_DIR}"; exit 1; }
[[ -f "${OUTPUT_PATH}" ]] || { echo "Missing ${OUTPUT_PATH}"; exit 1; }
[[ -f "${RESULT_DIR}/BENCHMARK_EXIT_0" ]] || { echo "Benchmark did not report exit 0"; exit 1; }

python - "${OUTPUT_PATH}" "${GPU_PROCESS_LOG}" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

result_path, process_path = map(Path, sys.argv[1:])
payload = json.loads(result_path.read_text())
expected = {
    "llama124m_adamw-spectral-l1-reg_wd1e-1_lr5e-3_sl1_0.01_finewebedu_erank",
    "llama257m_adamw-spectral-l1-reg_wd1e-1_lr3e-3_sl1_0.01_finewebedu_erank",
    "adamw_lr1e-3_finewebedu",
}
by_label = defaultdict(list)
for row in payload.get("checkpoints", []):
    label = Path(row["checkpoint"]).parents[2].name
    by_label[label].append(row)
    if row.get("factor_order_check", {}).get("factor_order") != "B_then_A":
        raise SystemExit(f"Factor order failed for {label}, margin {row.get('margin')}")
    matches = [x for x in row.get("comparisons", []) if x.get("batch_size") == 256]
    if len(matches) != 1 or matches[0].get("speedup", 0) <= 0:
        raise SystemExit(f"Missing batch-256 speed result for {label}, margin {row.get('margin')}")
if set(by_label) != expected:
    raise SystemExit(f"Unexpected completed labels: {sorted(by_label)}")
for label, rows in by_label.items():
    if rows[0].get("margin") != 0 or max(rows[-1].get("retained_ranks", {}).values(), default=2) != 1:
        raise SystemExit(f"Incomplete rank-to-one sweep for {label}")
if process_path.is_file():
    pids = set()
    for row in csv.reader(process_path.read_text().splitlines()):
        if len(row) >= 2 and row[1].strip().isdigit():
            pids.add(row[1].strip())
    if len(pids) != 1:
        raise SystemExit(f"Isolation watchdog saw {len(pids)} compute PIDs: {sorted(pids)}")
print(f"RESULT_VALIDATED_ROWS={sum(map(len, by_label.values()))}")
print("ISOLATION_WATCHDOG=OK")
PY

rm -rf "${CHECKPOINT_ROOT}"
rm -rf "${HF_CACHE_DIR}"
[[ ! -e "${CHECKPOINT_ROOT}" && ! -e "${HF_CACHE_DIR}" ]] || { echo "Cleanup verification failed"; exit 1; }
echo "CLOUD_L2_WD_CHECKPOINT_CLEANUP=OK"
echo "RESULT_PRESERVED=${OUTPUT_PATH}"
