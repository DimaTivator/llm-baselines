#!/usr/bin/env bash

set -euo pipefail

RESULT_DIR=${RESULT_DIR:-/home/jovyan/results/svdllm-inference-compile-cf1}

python - "${RESULT_DIR}" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path


result_dir = Path(sys.argv[1])
if not result_dir.is_dir():
    raise SystemExit(f"Missing result directory: {result_dir}")

for result_path in sorted(result_dir.glob("results-*-auto-floor-m256.json")):
    payload = json.loads(result_path.read_text())
    print(f"RESULT={result_path}")
    print(json.dumps(payload, separators=(",", ":")))

for process_path in sorted(result_dir.glob("gpu-processes-*-auto-floor-m256.csv")):
    samples = 0
    processes = Counter()
    for line in process_path.read_text().splitlines():
        if line.endswith("Z") and "T" in line:
            samples += 1
            continue
        row = next(csv.reader([line], skipinitialspace=True))
        if len(row) >= 2:
            processes[(row[0].strip(), row[1].strip())] += 1
    print(f"GPU_PROCESS_LOG={process_path}")
    print(f"GPU_SAMPLES={samples}")
    for (pid, process_name), count in sorted(processes.items()):
        print(f"GPU_PROCESS pid={pid} name={process_name} samples={count}")
PY
