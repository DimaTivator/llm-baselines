#!/usr/bin/env bash

set -euo pipefail

RESULT_DIR="${RESULT_DIR:?RESULT_DIR is required}"
OUTPUT_PATH="${RESULT_DIR}/results.json"
GPU_PROCESS_LOG="${RESULT_DIR}/gpu-processes.csv"

python - "${OUTPUT_PATH}" "${GPU_PROCESS_LOG}" <<'PY'
import csv
import json
import time
import sys
from collections import Counter
from pathlib import Path

result_path, process_path = map(Path, sys.argv[1:])
if not result_path.is_file():
    print("RESULT_PRESENT=0")
    raise SystemExit(0)

payload = json.loads(result_path.read_text())
entries = payload.get("checkpoints", [])
counts = Counter()
for entry in entries:
    counts[Path(entry["checkpoint"]).parents[2].name] += 1

print("RESULT_PRESENT=1")
print(f"RESULT_ROWS={len(entries)}")
print(f"RESULT_AGE_SECONDS={time.time() - result_path.stat().st_mtime:.1f}")
print(f"COMPILE_MODE={payload.get('compile_mode')}")
print(f"AUTO_RANK_MULTIPLE={payload.get('auto_rank_multiple')}")
print(f"RESIDUAL_GUARD={payload.get('max_whitened_relative_residual')}")
for label, count in sorted(counts.items()):
    print(f"MODEL_ROWS={label}:{count}")
if entries:
    last = entries[-1]
    last_label = Path(last["checkpoint"]).parents[2].name
    ranks = last.get("retained_ranks", {})
    print(f"LAST_MODEL={last_label}")
    print(f"LAST_MARGIN={last.get('margin')}")
    print(f"LAST_MIN_RANK={min(ranks.values()) if ranks else 'none'}")
    print(f"LAST_MAX_RANK={max(ranks.values()) if ranks else 'none'}")

pids = set()
if process_path.is_file():
    for row in csv.reader(process_path.read_text().splitlines()):
        if len(row) >= 2 and row[1].strip().isdigit():
            pids.add(row[1].strip())
print(f"WATCHDOG_PIDS={len(pids)}")
print(f"BENCHMARK_EXIT_MARKER={int((result_path.parent / 'BENCHMARK_EXIT_0').is_file())}")
PY
