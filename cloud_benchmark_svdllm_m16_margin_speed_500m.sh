#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?CHECKPOINT_ROOT is required}"
EXPERIMENT_LIST="${EXPERIMENT_LIST:-src/compression/svdllm_500m_speed_checkpoints.txt}"
EXPECTED_MODELS="${EXPECTED_MODELS:-6}"
RESULT_DIR="${RESULT_DIR:?RESULT_DIR is required}"
LOG_DIR="${LOG_DIR:-/tmp/svdllm-500m-margin-speed-logs}"
COMPILE_MODE="${COMPILE_MODE:-max-autotune}"
AUTO_RANK_MULTIPLE="${AUTO_RANK_MULTIPLE:-16}"
MARGINS="${MARGINS:-0 -10 10 -15 15 -20 20 -25 25 -30 30}"
MAX_RESUME_ATTEMPTS="${MAX_RESUME_ATTEMPTS:-8}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
OUTPUT_PATH="${RESULT_DIR}/results.json"
GPU_PROCESS_LOG="${RESULT_DIR}/gpu-processes.csv"
EXIT_MARKER="${RESULT_DIR}/BENCHMARK_EXIT_0"
LOG_PATH="${LOG_DIR}/benchmark-svdllm-500m-margin-speed-$(date +%F_%H%M%S).log"

export HOME="${COMPILE_HOME:-/tmp/svdllm-500m-margin-speed-home}"
export XDG_CACHE_HOME="${HOME}/.cache"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-svdllm-500m-margin-speed}"
export TRITON_HOME="${HOME}/.triton"
export TRITON_CACHE_DIR="${TRITON_HOME}/cache"
export PYTORCH_TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
mkdir -p "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

mapfile -t EXPERIMENTS < <(sed '/^[[:space:]]*$/d' "${EXPERIMENT_LIST}")
if [[ "${#EXPERIMENTS[@]}" -ne "${EXPECTED_MODELS}" ]]; then
    echo "Expected ${EXPECTED_MODELS} experiments, found ${#EXPERIMENTS[@]}"
    exit 1
fi

CHECKPOINTS=()
for experiment in "${EXPERIMENTS[@]}"; do
    checkpoint="${CHECKPOINT_ROOT}/${experiment}/ckpts/latest/main.pt"
    [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}"; exit 1; }
    [[ -f "${CHECKPOINT_ROOT}/${experiment}/summary.json" ]] || {
        echo "Missing summary: ${CHECKPOINT_ROOT}/${experiment}/summary.json"; exit 1;
    }
    CHECKPOINTS+=("${checkpoint}")
done

CALIBRATION="${CHECKPOINT_ROOT}/calibration/val.bin"
[[ -f "${CALIBRATION}" ]] || { echo "Missing calibration tokens: ${CALIBRATION}"; exit 1; }
read -r -a MARGIN_ARGS <<<"${MARGINS//,/ }"
[[ "${#MARGIN_ARGS[@]}" -eq 11 ]] || { echo "Expected 11 margins, found ${#MARGIN_ARGS[@]}"; exit 1; }
[[ "${AUTO_RANK_MULTIPLE}" == "16" ]] || { echo "Expected AUTO_RANK_MULTIPLE=16"; exit 1; }
[[ "${COMPILE_MODE}" == "max-autotune" ]] || { echo "Expected COMPILE_MODE=max-autotune"; exit 1; }

result_rows() {
    python - "${OUTPUT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(len(json.loads(path.read_text()).get("checkpoints", [])) if path.is_file() else 0)
PY
}

run_with_resume() {
    local attempt before_rows after_rows status
    for ((attempt = 1; attempt <= MAX_RESUME_ATTEMPTS; attempt++)); do
        before_rows="$(result_rows)"
        echo "BENCHMARK_ATTEMPT=${attempt} BEFORE_ROWS=${before_rows}"
        set +e
        PYTHONUNBUFFERED=1 PYTHONPATH=src python src/compression/benchmark_svd_llm_inference.py \
            "${CHECKPOINTS[@]}" \
            --device cuda:0 --dtype bfloat16 \
            --compile_mode "${COMPILE_MODE}" --disable_inductor_pattern_matcher \
            --auto_rank_multiple "${AUTO_RANK_MULTIPLE}" \
            --margins "${MARGIN_ARGS[@]}" \
            --batch_sizes 256 --calib_batches 16 --calib_batch_size 8 \
            --warmup_steps 10 --timed_steps 50 --calibration_tokens "${CALIBRATION}" \
            --output "${OUTPUT_PATH}"
        status=$?
        set -e
        after_rows="$(result_rows)"
        echo "BENCHMARK_ATTEMPT_EXIT=${status} AFTER_ROWS=${after_rows}"
        if [[ "${status}" -eq 0 ]]; then
            return 0
        fi
        if [[ "${after_rows}" -le "${before_rows}" ]]; then
            echo "Benchmark failed without progress; refusing another automatic retry"
            return "${status}"
        fi
        echo "Benchmark made progress before failure; resuming in a fresh Python process"
    done
    echo "Exhausted ${MAX_RESUME_ATTEMPTS} benchmark attempts"
    return 1
}

monitor_gpu_processes() {
    local worker_pid=$1
    while kill -0 "${worker_pid}" 2>/dev/null; do
        date -u +%Y-%m-%dT%H:%M:%SZ >>"${GPU_PROCESS_LOG}"
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader >>"${GPU_PROCESS_LOG}"
        sleep 2
    done
}

rm -f "${EXIT_MARKER}"
: >"${GPU_PROCESS_LOG}"
set -o pipefail
(
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
    [[ "${GPU_NAME}" == *H100* ]] || { echo "Expected H100, found ${GPU_NAME}"; exit 1; }
    INITIAL_PROCESSES="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)"
    [[ "${INITIAL_PROCESSES}" -eq 0 ]] || {
        echo "GPU not isolated: ${INITIAL_PROCESSES} compute processes"; exit 1;
    }
    nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,memory.free --format=csv
    run_with_resume &
    WORKER_PID=$!
    monitor_gpu_processes "${WORKER_PID}" &
    MONITOR_PID=$!
    set +e
    wait "${WORKER_PID}"; BENCHMARK_STATUS=$?
    wait "${MONITOR_PID}"
    set -e
    echo "BENCHMARK_EXIT=${BENCHMARK_STATUS}"
    exit "${BENCHMARK_STATUS}"
) 2>&1 | tee "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}

if [[ "${STATUS}" -eq 0 ]]; then
    python - "${OUTPUT_PATH}" "$((EXPECTED_MODELS * ${#MARGIN_ARGS[@]}))" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
actual = len(payload.get("checkpoints", []))
if actual != expected:
    raise SystemExit(f"Expected {expected} rows, found {actual}")
print(f"RESULT_ROWS={actual}")
PY
    : >"${EXIT_MARKER}"
fi

echo "EXIT=${STATUS}"
echo "LOG=${LOG_PATH}"
echo "RESULT=${OUTPUT_PATH}"
echo "GPU_PROCESS_LOG=${GPU_PROCESS_LOG}"
exit "${STATUS}"
