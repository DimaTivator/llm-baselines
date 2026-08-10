#!/bin/bash

set -uo pipefail

RESULT_DIR=/home/jovyan/results/fused-low-rank-cf1
mkdir -p "$RESULT_DIR"
LOG_PATH="$RESULT_DIR/run.log"
RESULT_PATH="$RESULT_DIR/microbenchmark.json"
GPU_PROCESS_LOG="$RESULT_DIR/gpu-processes.csv"

if [[ "${1:-}" == "cpu-smoke" ]]; then
  {
    python -m py_compile \
      src/models/fused_low_rank.py \
      src/models/compress.py \
      src/compression/svd_llm.py \
      src/compression/benchmark_fused_low_rank.py
    PYTHONPATH=src python - <<'PY'
import torch
import torch.nn.functional as F

from models.compress import LowRankLinear
from models.fused_low_rank import fused_low_rank_linear

x = torch.randn(3, 8)
b_weight = torch.randn(4, 8)
a_weight = torch.randn(12, 4)
bias = torch.randn(12)
expected = F.linear(F.linear(x, b_weight), a_weight, bias)
actual = fused_low_rank_linear(x, b_weight, a_weight, bias)
torch.testing.assert_close(actual, expected)

try:
    LowRankLinear(8, 12, 4, kernel="unknown")
except ValueError:
    pass
else:
    raise AssertionError("LowRankLinear accepted an unknown kernel")

print("CPU fallback correctness: OK")
PY
    echo "CPU_SMOKE_EXIT=$?"
  } 2>&1 | tee "$LOG_PATH"
  exit 0
fi

if [[ "${1:-}" == "report-results" ]]; then
  python - <<'PY'
import json
from pathlib import Path

result_dir = Path("/home/jovyan/results/fused-low-rank-cf1")
for path in sorted(result_dir.glob("*.json")):
    payload = json.loads(path.read_text())
    print(f"=== {path.name} ===")
    print(json.dumps(payload.get("aggregates", []), indent=2))

process_log = result_dir / "gpu-processes.csv"
process_rows = []
if process_log.exists():
    for line in process_log.read_text().splitlines():
        if line and not line[0].isdigit():
            process_rows.append(line)
        elif "," in line:
            process_rows.append(line)
print("=== unique GPU process rows ===")
for row in sorted(set(process_rows)):
    print(row)
PY
  exit 0
fi

monitor_gpu_processes() {
  local benchmark_pid=$1
  while kill -0 "$benchmark_pid" 2>/dev/null; do
    date -u +%Y-%m-%dT%H:%M:%SZ >> "$GPU_PROCESS_LOG"
    nvidia-smi \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader >> "$GPU_PROCESS_LOG"
    sleep 2
  done
}

{
  echo "=== GPU before benchmark ==="
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  echo "=== benchmark ==="
} 2>&1 | tee "$LOG_PATH"

: > "$GPU_PROCESS_LOG"

PYTHONPATH=src python src/compression/benchmark_fused_low_rank.py \
  --warmup_steps 10 \
  --timed_steps 50 \
  --output "$RESULT_PATH" \
  "$@" \
  > >(tee -a "$LOG_PATH") 2>&1 &
BENCHMARK_PID=$!
monitor_gpu_processes "$BENCHMARK_PID" &
MONITOR_PID=$!
wait "$BENCHMARK_PID"
BENCHMARK_EXIT=$?
wait "$MONITOR_PID"

{
  echo "=== GPU after benchmark ==="
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  echo "GPU_PROCESS_LOG=$GPU_PROCESS_LOG"
  echo "BENCHMARK_EXIT=$BENCHMARK_EXIT"
} 2>&1 | tee -a "$LOG_PATH"

exit 0
