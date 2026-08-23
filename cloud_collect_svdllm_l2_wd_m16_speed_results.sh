#!/usr/bin/env bash

set -euo pipefail

RESULT_DIR="${RESULT_DIR:?RESULT_DIR is required}"
OUTPUT_PATH="${RESULT_DIR}/results.json"
GPU_PROCESS_LOG="${RESULT_DIR}/gpu-processes.csv"
EXPECTED_ROWS="${EXPECTED_ROWS:-87}"
EXPECTED_RESIDUAL_GUARD="${EXPECTED_RESIDUAL_GUARD:-none}"
[[ -f "${OUTPUT_PATH}" ]] || { echo "Missing ${OUTPUT_PATH}"; exit 1; }
[[ -f "${RESULT_DIR}/BENCHMARK_EXIT_0" ]] || { echo "BENCHMARK_EXIT_0 marker missing"; exit 1; }

python - "${OUTPUT_PATH}" "${GPU_PROCESS_LOG}" "${EXPECTED_ROWS}" "${EXPECTED_RESIDUAL_GUARD}" <<'PY'
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

result_path, process_path, expected_rows = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
expected_guard = None if sys.argv[4] == "none" else float(sys.argv[4])
payload = json.loads(result_path.read_text())
expected_labels = {
    "llama124m_adamw-spectral-l1-reg_wd1e-1_lr5e-3_sl1_0.01_finewebedu_erank",
    "llama257m_adamw-spectral-l1-reg_wd1e-1_lr3e-3_sl1_0.01_finewebedu_erank",
    "adamw_lr1e-3_finewebedu",
}
if payload.get("compile_mode") != "max-autotune":
    raise SystemExit(f"Expected max-autotune, found {payload.get('compile_mode')}")
if payload.get("auto_rank_multiple") != 16:
    raise SystemExit(f"Expected auto_rank_multiple=16, found {payload.get('auto_rank_multiple')}")
if payload.get("max_whitened_relative_residual") != expected_guard:
    raise SystemExit(
        "Wrong whitened residual guard: "
        f"expected {expected_guard}, found {payload.get('max_whitened_relative_residual')}"
    )
entries = payload.get("checkpoints", [])
if len(entries) != expected_rows:
    raise SystemExit(f"Expected {expected_rows} rows, found {len(entries)}")
rows, by_label = [], defaultdict(list)
for entry in entries:
    label = Path(entry["checkpoint"]).parents[2].name
    comparison = next((x for x in entry["comparisons"] if x.get("batch_size") == 256), None)
    if comparison is None or entry.get("factor_order_check", {}).get("factor_order") != "B_then_A":
        raise SystemExit(f"Invalid factor-order or batch-256 row: {label}, margin={entry.get('margin')}")
    ranks = entry["retained_ranks"]
    if not ranks or any(rank % 16 for rank in ranks.values()):
        raise SystemExit(f"Non-m16 retained rank: {label}, margin={entry.get('margin')}")
    dense = next(x for x in entry["measurements"] if x["model"] == "original" and x["batch_size"] == 256)
    compressed = next(x for x in entry["measurements"] if x["model"] == "compressed" and x["batch_size"] == 256)
    row = {
        "label": label,
        "margin": entry["margin"],
        "compression_ratio": entry["parameter_compression_ratio"],
        "mean_rank": sum(ranks.values()) / len(ranks),
        "min_rank": min(ranks.values()),
        "dense_latency_ms": dense["latency_ms"],
        "compressed_latency_ms": compressed["latency_ms"],
        "speedup": comparison["speedup"],
    }
    rows.append(row)
    by_label[label].append(row)
if set(by_label) != expected_labels:
    raise SystemExit(f"Wrong model labels: {sorted(by_label)}")
for label, values in by_label.items():
    if values[0]["margin"] != 0:
        raise SystemExit(f"Missing margin 0: {label}")
pids = {
    row[1].strip()
    for row in csv.reader(process_path.read_text().splitlines())
    if len(row) >= 2 and row[1].strip().isdigit()
}
if len(pids) != 1:
    raise SystemExit(f"Isolation watchdog saw {len(pids)} PIDs")
print("BENCHMARK_EXIT=0")
print("ISOLATION_WATCHDOG=OK")
print(f"RESULT_ROWS={len(rows)}")
print(f"RESULT_SHA256={hashlib.sha256(result_path.read_bytes()).hexdigest()}")
print("COMPACT_RESULTS_JSON=" + json.dumps(rows, separators=(",", ":")))
PY
