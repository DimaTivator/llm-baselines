#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace-SR006.nfs2/dimativator/svdllm-l2-wd-speed-20260822}"
RESULT_DIR="${RESULT_DIR:-/workspace-SR006.nfs2/dimativator/results/svdllm-l2-wd-speed-20260822}"
EXPECTED_ROOT="${EXPECTED_ROOT:-/workspace-SR006.nfs2/dimativator/svdllm-l2-wd-speed-20260822}"
EXPECTED_RESULT="${EXPECTED_RESULT:-/workspace-SR006.nfs2/dimativator/results/svdllm-l2-wd-speed-20260822}"
HF_CACHE_DIR="/workspace-SR006.nfs2/dimativator/.hf-cache/datasets--DimaTivator--svdllm-l2-wd-speed-checkpoints"
EXPECT_RANK_ONE="${EXPECT_RANK_ONE:-1}"
EXPECTED_AUTO_RANK_MULTIPLE="${EXPECTED_AUTO_RANK_MULTIPLE:-}"
EXPECTED_COMPILE_MODE="${EXPECTED_COMPILE_MODE:-}"
EXPECTED_ROWS="${EXPECTED_ROWS:-}"
EXPECTED_RESIDUAL_GUARD="${EXPECTED_RESIDUAL_GUARD:-}"
OUTPUT_PATH="${RESULT_DIR}/results.json"
GPU_PROCESS_LOG="${RESULT_DIR}/gpu-processes.csv"

[[ "${EXPECTED_ROOT}" =~ ^/workspace-SR006\.nfs2/dimativator/svdllm-l2-wd-speed-[0-9]{8}$ ]] || {
    echo "Refusing unsafe EXPECTED_ROOT=${EXPECTED_ROOT}"; exit 1;
}
[[ "$(realpath -m "${CHECKPOINT_ROOT}")" == "${EXPECTED_ROOT}" ]] || { echo "Refusing CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"; exit 1; }
[[ "$(realpath -m "${RESULT_DIR}")" == "${EXPECTED_RESULT}" ]] || { echo "Refusing RESULT_DIR=${RESULT_DIR}"; exit 1; }
[[ -f "${OUTPUT_PATH}" ]] || { echo "Missing ${OUTPUT_PATH}"; exit 1; }
[[ -f "${RESULT_DIR}/BENCHMARK_EXIT_0" ]] || { echo "Benchmark did not report exit 0"; exit 1; }

python - "${OUTPUT_PATH}" "${GPU_PROCESS_LOG}" "${EXPECT_RANK_ONE}" "${EXPECTED_AUTO_RANK_MULTIPLE}" "${EXPECTED_COMPILE_MODE}" "${EXPECTED_ROWS}" "${EXPECTED_RESIDUAL_GUARD}" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

result_path, process_path = map(Path, sys.argv[1:3])
expect_rank_one = sys.argv[3] == "1"
expected_multiple = None if not sys.argv[4] else int(sys.argv[4])
expected_compile_mode = sys.argv[5] or None
expected_rows = None if not sys.argv[6] else int(sys.argv[6])
expected_guard = None if sys.argv[7] == "none" else (float(sys.argv[7]) if sys.argv[7] else "unchecked")
payload = json.loads(result_path.read_text())
expected = {
    "llama124m_adamw-spectral-l1-reg_wd1e-1_lr5e-3_sl1_0.01_finewebedu_erank",
    "llama257m_adamw-spectral-l1-reg_wd1e-1_lr3e-3_sl1_0.01_finewebedu_erank",
    "adamw_lr1e-3_finewebedu",
}
by_label = defaultdict(list)
if expected_multiple is not None and payload.get("auto_rank_multiple") != expected_multiple:
    raise SystemExit(f"Expected auto_rank_multiple={expected_multiple}, found {payload.get('auto_rank_multiple')}")
if expected_compile_mode is not None and payload.get("compile_mode") != expected_compile_mode:
    raise SystemExit(f"Expected compile_mode={expected_compile_mode}, found {payload.get('compile_mode')}")
if expected_rows is not None and len(payload.get("checkpoints", [])) != expected_rows:
    raise SystemExit(f"Expected {expected_rows} rows, found {len(payload.get('checkpoints', []))}")
if expected_guard != "unchecked" and payload.get("max_whitened_relative_residual") != expected_guard:
    raise SystemExit(
        f"Expected residual guard {expected_guard}, "
        f"found {payload.get('max_whitened_relative_residual')}"
    )
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
    if rows[0].get("margin") != 0:
        raise SystemExit(f"Missing margin 0 for {label}")
    if expect_rank_one and max(rows[-1].get("retained_ranks", {}).values(), default=2) != 1:
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
