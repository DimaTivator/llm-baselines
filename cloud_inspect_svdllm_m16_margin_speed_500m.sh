#!/usr/bin/env bash

set -euo pipefail

RESULT_DIR="${RESULT_DIR:?RESULT_DIR is required}"
OUTPUT_PATH="${RESULT_DIR}/results.json"

python - "${OUTPUT_PATH}" <<'PY'
import json
import time
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("RESULT_PRESENT=0")
    raise SystemExit(0)
payload = json.loads(path.read_text())
entries = payload.get("checkpoints", [])
counts = Counter(Path(row["checkpoint"]).parents[2].name for row in entries)
print("RESULT_PRESENT=1")
print(f"RESULT_ROWS={len(entries)}")
print(f"RESULT_AGE_SECONDS={time.time() - path.stat().st_mtime:.1f}")
print(f"COMPILE_MODE={payload.get('compile_mode')}")
print(f"AUTO_RANK_MULTIPLE={payload.get('auto_rank_multiple')}")
print(f"RESIDUAL_GUARD={payload.get('max_whitened_relative_residual')}")
for label, count in sorted(counts.items()):
    print(f"MODEL_ROWS={label}:{count}")
if entries:
    last = entries[-1]
    ranks = last.get("retained_ranks", {})
    print(f"LAST_MODEL={Path(last['checkpoint']).parents[2].name}")
    print(f"LAST_MARGIN={last.get('margin')}")
    print(f"LAST_MIN_RANK={min(ranks.values()) if ranks else 'none'}")
    print(f"LAST_MAX_RANK={max(ranks.values()) if ranks else 'none'}")
print(f"BENCHMARK_EXIT_MARKER={int((path.parent / 'BENCHMARK_EXIT_0').is_file())}")
PY
