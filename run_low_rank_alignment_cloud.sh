#!/bin/bash

set -uo pipefail

RESULT_DIR=/home/jovyan/results/low-rank-alignment-cf1
mkdir -p "$RESULT_DIR"
LOG_PATH="$RESULT_DIR/run.log"
RESULT_PATH="$RESULT_DIR/microbenchmark.json"
GPU_PROCESS_LOG="$RESULT_DIR/gpu-processes.csv"

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
PYTHONPATH=src python src/compression/benchmark_low_rank_alignment.py \
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
