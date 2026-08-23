#!/usr/bin/env bash

set -euo pipefail

RESULT_DIR="${RESULT_DIR:?RESULT_DIR is required}"
OUTPUT_PATH="${RESULT_DIR}/results.json"
GPU_PROCESS_LOG="${RESULT_DIR}/gpu-processes.csv"
EXPECTED_ROWS="${EXPECTED_ROWS:-66}"
EXPECTED_GUARD="${EXPECTED_GUARD:-0.05}"

[[ -f "${OUTPUT_PATH}" ]] || { echo "Missing ${OUTPUT_PATH}"; exit 1; }
[[ -f "${GPU_PROCESS_LOG}" ]] || { echo "Missing ${GPU_PROCESS_LOG}"; exit 1; }
[[ -f "${RESULT_DIR}/BENCHMARK_EXIT_0" ]] || { echo "BENCHMARK_EXIT_0 marker missing"; exit 1; }

python - "${OUTPUT_PATH}" "${GPU_PROCESS_LOG}" "${EXPECTED_ROWS}" "${EXPECTED_GUARD}" <<'PY'
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

result_path = Path(sys.argv[1])
process_path = Path(sys.argv[2])
expected_rows = int(sys.argv[3])
expected_guard = float(sys.argv[4])
payload = json.loads(result_path.read_text())
expected_labels = {
    "h200_gpu2_05ch_bs32acc4_llama500m_adamw-spectral-l1-reg_wd1e-1_lr1e-3_sl1_0.5_finewebedu_erank",
    "h200_gpu2_05ch_bs32acc4_llama500m_adamw-spectral-l1-reg_wd1e-1_lr1e-3_sl1_0.7_finewebedu_erank",
    "h200_gpu2_05ch_bs32acc4_llama500m_adamw-spectral-l1-reg_wd1e-1_lr1e-3_sl1_1_finewebedu_erank",
    "h200_gpu4_05ch_bs32acc4_llama500m_adamw-spectral-l1-reg_wd1e-1_lr1e-3_sl1_1.2_finewebedu_erank",
    "h200_gpu4_05ch_bs32acc4_llama500m_adamw-spectral-l1-reg_wd1e-1_lr1e-3_sl1_1.4_finewebedu_erank",
    "h200_gpu4_05ch_bs32acc4_llama500m_adamw-spectral-l1-reg_wd1e-1_lr1e-3_sl1_1.6_finewebedu_erank",
}
expected_margins = {0, -10, 10, -15, 15, -20, 20, -25, 25, -30, 30}
if payload.get("compile_mode") != "max-autotune":
    raise SystemExit(f"Wrong compile mode: {payload.get('compile_mode')}")
if payload.get("auto_rank_multiple") != 16:
    raise SystemExit(f"Wrong auto_rank_multiple: {payload.get('auto_rank_multiple')}")
if payload.get("max_whitened_relative_residual") != expected_guard:
    raise SystemExit(
        f"Wrong residual guard: expected {expected_guard}, "
        f"found {payload.get('max_whitened_relative_residual')}"
    )
entries = payload.get("checkpoints", [])
if len(entries) != expected_rows:
    raise SystemExit(f"Expected {expected_rows} rows, found {len(entries)}")

rows = []
by_label = defaultdict(list)
for entry in entries:
    label = Path(entry["checkpoint"]).parents[2].name
    margin = int(entry["margin"])
    comparison = next((x for x in entry["comparisons"] if x.get("batch_size") == 256), None)
    if comparison is None:
        raise SystemExit(f"Missing batch-256 comparison: {label}, margin={margin}")
    if entry.get("factor_order_check", {}).get("factor_order") != "B_then_A":
        raise SystemExit(f"Wrong factor order: {label}, margin={margin}")
    ranks = entry.get("retained_ranks", {})
    if not ranks or any(rank % 16 for rank in ranks.values()):
        raise SystemExit(f"Non-m16 rank: {label}, margin={margin}")
    dense = next(x for x in entry["measurements"] if x["model"] == "original" and x["batch_size"] == 256)
    compressed = next(x for x in entry["measurements"] if x["model"] == "compressed" and x["batch_size"] == 256)
    row = {
        "label": label,
        "margin": margin,
        "compression_ratio": entry["parameter_compression_ratio"],
        "mean_rank": sum(ranks.values()) / len(ranks),
        "min_rank": min(ranks.values()),
        "max_rank": max(ranks.values()),
        "dense_latency_ms": dense["latency_ms"],
        "compressed_latency_ms": compressed["latency_ms"],
        "speedup": comparison["speedup"],
    }
    rows.append(row)
    by_label[label].append(row)
if set(by_label) != expected_labels:
    raise SystemExit(f"Wrong labels: {sorted(by_label)}")
for label, values in by_label.items():
    if {row["margin"] for row in values} != expected_margins:
        raise SystemExit(f"Wrong margins for {label}")

pids = set()
max_concurrent = 0
current_group = 0
timestamp = re.compile(r"^\d{4}-\d{2}-\d{2}T")
for raw in process_path.read_text().splitlines():
    if timestamp.match(raw):
        max_concurrent = max(max_concurrent, current_group)
        current_group = 0
        continue
    parsed = next(csv.reader([raw]))
    if len(parsed) >= 2 and parsed[1].strip().isdigit():
        current_group += 1
        pids.add(parsed[1].strip())
max_concurrent = max(max_concurrent, current_group)
if not pids or max_concurrent != 1:
    raise SystemExit(
        f"Isolation failure: pids={sorted(pids)}, max_concurrent={max_concurrent}"
    )

print("BENCHMARK_EXIT=0")
print("ISOLATION_WATCHDOG=OK")
print(f"WATCHDOG_UNIQUE_PIDS={len(pids)}")
print(f"RESULT_ROWS={len(rows)}")
print(f"RESULT_SHA256={hashlib.sha256(result_path.read_bytes()).hexdigest()}")
print("COMPACT_RESULTS_JSON=" + json.dumps(rows, separators=(",", ":")))
PY
