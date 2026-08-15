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
    checkpoint = payload["checkpoints"][0]
    rank_counts = Counter(checkpoint["retained_ranks"].values())
    print(
        f"META mode={payload['compile_mode']} "
        f"original_params={checkpoint['original_parameters']} "
        f"compressed_params={checkpoint['compressed_parameters']} "
        f"compression={checkpoint['parameter_compression_ratio']:.6f} "
        f"original_max_batch={checkpoint['original_max_batch_size']} "
        f"compressed_max_batch={checkpoint['compressed_max_batch_size']} "
        f"rank_counts={dict(sorted(rank_counts.items()))} "
        f"factor_error={checkpoint['factor_order_check']['max_abs_error']}"
    )
    measurements = {
        (row["model"], row["batch_size"]): row
        for row in checkpoint["measurements"]
    }
    for comparison in checkpoint["comparisons"]:
        batch_size = comparison["batch_size"]
        original = measurements[("original", batch_size)]
        compressed = measurements[("compressed", batch_size)]
        print(
            f"ROW mode={payload['compile_mode']} batch={batch_size} "
            f"original_ms={original['latency_ms']:.6f} "
            f"compressed_ms={compressed['latency_ms']:.6f} "
            f"speedup={comparison['speedup']:.6f} "
            f"original_peak_mib={original['peak_memory_mib']:.3f} "
            f"compressed_peak_mib={compressed['peak_memory_mib']:.3f}"
        )

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
