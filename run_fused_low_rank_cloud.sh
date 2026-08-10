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
    PYTHONPATH=src python -m pytest -q tests/test_fused_low_rank.py
    echo "CPU_SMOKE_EXIT=$?"
  } 2>&1 | tee "$LOG_PATH"
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

PYTHONPATH=src python src/compression/benchmark_fused_low_rank.py \
  --warmup_steps 10 \
  --timed_steps 50 \
  --output "$RESULT_PATH" \
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
