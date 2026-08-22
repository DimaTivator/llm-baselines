#!/usr/bin/env bash

set -euo pipefail

RESULT_DIR="${RESULT_DIR:-/workspace-SR006.nfs2/dimativator/results/svdllm-l2-wd-speed-reduce-overhead-20260822}"
OUTPUT_PATH="${RESULT_DIR}/results.json"
if [[ ! -f "${OUTPUT_PATH}" ]]; then
    echo "RESULT_NOT_WRITTEN=1"
    exit 0
fi

python - "${OUTPUT_PATH}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

payload = json.load(open(sys.argv[1]))
by_label = defaultdict(list)
for entry in payload.get("checkpoints", []):
    label = Path(entry["checkpoint"]).parents[2].name
    comp = next((x for x in entry.get("comparisons", []) if x.get("batch_size") == 256), None)
    by_label[label].append((entry.get("margin"), None if comp is None else comp.get("speedup")))
print(f"RESULT_ROWS={sum(map(len, by_label.values()))}")
for label, rows in sorted(by_label.items()):
    margin, speedup = rows[-1]
    print(f"PROGRESS label={label} rows={len(rows)} last_margin={margin} last_speedup={speedup}")
PY
