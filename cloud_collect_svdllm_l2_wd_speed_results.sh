#!/usr/bin/env bash

set -euo pipefail

RESULT_DIR="${RESULT_DIR:-/workspace-SR006.nfs2/dimativator/results/svdllm-l2-wd-speed-20260822}"
OUTPUT_PATH="${RESULT_DIR}/results.json"
GPU_PROCESS_LOG="${RESULT_DIR}/gpu-processes.csv"
[[ -f "${OUTPUT_PATH}" ]] || { echo "Missing ${OUTPUT_PATH}"; exit 1; }
[[ -f "${RESULT_DIR}/BENCHMARK_EXIT_0" ]] || { echo "BENCHMARK_EXIT_0 marker missing"; exit 1; }

python - "${OUTPUT_PATH}" "${GPU_PROCESS_LOG}" <<'PY'
import csv
import hashlib
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
rows = []
by_label = defaultdict(list)
for entry in payload.get("checkpoints", []):
    label = Path(entry["checkpoint"]).parents[2].name
    comparison = next((x for x in entry.get("comparisons", []) if x.get("batch_size") == 256), None)
    dense = next(x for x in entry["measurements"] if x["model"] == "original" and x["batch_size"] == 256)
    compressed = next(x for x in entry["measurements"] if x["model"] == "compressed" and x["batch_size"] == 256)
    if entry.get("factor_order_check", {}).get("factor_order") != "B_then_A" or comparison is None:
        raise SystemExit(f"Invalid entry: {label}, margin={entry.get('margin')}")
    compact = {
        "label": label,
        "margin": entry["margin"],
        "compression_ratio": entry["parameter_compression_ratio"],
        "mean_rank": sum(entry["retained_ranks"].values()) / len(entry["retained_ranks"]),
        "min_rank": min(entry["retained_ranks"].values()),
        "dense_latency_ms": dense["latency_ms"],
        "compressed_latency_ms": compressed["latency_ms"],
        "speedup": comparison["speedup"],
        "dense_tokens_per_second": dense["tokens_per_second"],
        "compressed_tokens_per_second": compressed["tokens_per_second"],
    }
    rows.append(compact)
    by_label[label].append(compact)
if set(by_label) != expected:
    raise SystemExit(f"Wrong model labels: {sorted(by_label)}")
for label, entries in by_label.items():
    if entries[0]["margin"] != 0 or entries[-1]["min_rank"] != 1:
        raise SystemExit(f"Incomplete rank-to-one sweep: {label}")
pids = set()
for row in csv.reader(process_path.read_text().splitlines()):
    if len(row) >= 2 and row[1].strip().isdigit():
        pids.add(row[1].strip())
if len(pids) != 1:
    raise SystemExit(f"Isolation watchdog saw {len(pids)} PIDs")
print("BENCHMARK_EXIT=0")
print("ISOLATION_WATCHDOG=OK")
print(f"RESULT_SHA256={hashlib.sha256(result_path.read_bytes()).hexdigest()}")
print("COMPACT_RESULTS_JSON=" + json.dumps(rows, separators=(",", ":")))
PY
