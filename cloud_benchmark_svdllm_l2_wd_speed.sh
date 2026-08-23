#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace-SR006.nfs2/dimativator/svdllm-l2-wd-speed-20260822}"
EXPERIMENT_LIST="${EXPERIMENT_LIST:-src/compression/svdllm_l2_wd_speed_checkpoints.txt}"
EXPECTED_MODELS="${EXPECTED_MODELS:-3}"
RESULT_DIR="${RESULT_DIR:-/workspace-SR006.nfs2/dimativator/results/svdllm-l2-wd-speed-20260822}"
LOG_DIR="${LOG_DIR:-/home/jovyan/logs}"
COMPILE_MODE="${COMPILE_MODE:-max-autotune}"
AUTO_RANK_MULTIPLE="${AUTO_RANK_MULTIPLE:-}"
DISABLE_WHITENED_RESIDUAL_GUARD="${DISABLE_WHITENED_RESIDUAL_GUARD:-1}"
STOP_AT_RANK_ONE="${STOP_AT_RANK_ONE:-1}"
MARGINS="${MARGINS:-0 -10 -25 -50 -100 -150 -200 -250 -300 -350 -400 -450 -500 -550 -600 -650 -700 -750 -800 -850 -900 -950 -1000 -1050 -1100 -1150 -1200 -1250 -1300}"
MARGINS="${MARGINS//,/ }"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
OUTPUT_PATH="${RESULT_DIR}/results.json"
GPU_PROCESS_LOG="${RESULT_DIR}/gpu-processes.csv"
LOG_PATH="${LOG_DIR}/benchmark-svdllm-l2-wd-speed-$(date +%F_%H%M%S).log"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-svdllm-l2-wd-speed}"
# Recent Triton releases derive their cache from HOME/XDG_CACHE_HOME and can
# ignore TRITON_CACHE_DIR. /home/jovyan is quota-limited on Cloud.ru, whereas
# /tmp is local to this isolated benchmark pod and has enough room for
# max-autotune's generated kernels.
export HOME="${COMPILE_HOME:-/tmp/svdllm-l2-wd-home}"
export XDG_CACHE_HOME="${HOME}/.cache"
# Cloud.ru exports a pre-existing quota-limited TRITON_CACHE_DIR. Override it
# unconditionally rather than retaining the inherited value.
export TRITON_HOME="${HOME}/.triton"
export TRITON_CACHE_DIR="${TRITON_HOME}/cache"
export PYTORCH_TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}"

mapfile -t EXPERIMENTS < <(sed '/^[[:space:]]*$/d' "${EXPERIMENT_LIST}")
if [[ "${#EXPERIMENTS[@]}" -ne "${EXPECTED_MODELS}" ]]; then
    echo "Expected ${EXPECTED_MODELS} experiments, found ${#EXPERIMENTS[@]}"
    exit 1
fi
CHECKPOINTS=()
for experiment in "${EXPERIMENTS[@]}"; do
    checkpoint="${CHECKPOINT_ROOT}/${experiment}/ckpts/latest/main.pt"
    [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}"; exit 1; }
    CHECKPOINTS+=("${checkpoint}")
done
CALIBRATION="${CHECKPOINT_ROOT}/calibration/val.bin"
[[ -f "${CALIBRATION}" ]] || { echo "Missing calibration tokens: ${CALIBRATION}"; exit 1; }

COMPRESSION_ARGS=()
if [[ -n "${AUTO_RANK_MULTIPLE}" ]]; then
    [[ "${AUTO_RANK_MULTIPLE}" =~ ^[0-9]+$ ]] || {
        echo "AUTO_RANK_MULTIPLE must be an integer"; exit 1;
    }
    COMPRESSION_ARGS+=(--auto_rank_multiple "${AUTO_RANK_MULTIPLE}")
fi
if [[ "${DISABLE_WHITENED_RESIDUAL_GUARD}" == "1" ]]; then
    COMPRESSION_ARGS+=(--disable_whitened_residual_guard)
fi
if [[ "${STOP_AT_RANK_ONE}" == "1" ]]; then
    COMPRESSION_ARGS+=(--stop_at_rank_one)
fi
read -r -a MARGIN_ARGS <<<"${MARGINS}"
[[ "${#MARGIN_ARGS[@]}" -gt 0 ]] || { echo "MARGINS must not be empty"; exit 1; }

monitor_gpu_processes() {
    local benchmark_pid=$1
    while kill -0 "${benchmark_pid}" 2>/dev/null; do
        date -u +%Y-%m-%dT%H:%M:%SZ >>"${GPU_PROCESS_LOG}"
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >>"${GPU_PROCESS_LOG}"
        sleep 2
    done
}

set -o pipefail
(
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
    [[ "${GPU_NAME}" == *H100* ]] || { echo "Expected H100, found ${GPU_NAME}"; exit 1; }
    INITIAL_PROCESSES="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)"
    [[ "${INITIAL_PROCESSES}" -eq 0 ]] || { echo "GPU not isolated: ${INITIAL_PROCESSES} compute processes"; exit 1; }
    python - <<'PY'
import os

# ``triton.knobs`` is not available in the older Triton bundled with every
# Cloud.ru PyTorch image.  The benchmark only needs to verify the cache env;
# importing a version-specific diagnostic API must not abort the run.
print(f"TRITON_HOME={os.environ.get('TRITON_HOME')}", flush=True)
print(f"TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR')}", flush=True)
PY
    nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,memory.free --format=csv
    : >"${GPU_PROCESS_LOG}"
    PYTHONUNBUFFERED=1 PYTHONPATH=src python src/compression/benchmark_svd_llm_inference.py \
        "${CHECKPOINTS[@]}" \
        --device cuda:0 --dtype bfloat16 \
        --compile_mode "${COMPILE_MODE}" --disable_inductor_pattern_matcher \
        --margins "${MARGIN_ARGS[@]}" \
        "${COMPRESSION_ARGS[@]}" \
        --batch_sizes 256 --calib_batches 16 --calib_batch_size 8 \
        --warmup_steps 10 --timed_steps 50 --calibration_tokens "${CALIBRATION}" \
        --output "${OUTPUT_PATH}" &
    BENCHMARK_PID=$!
    monitor_gpu_processes "${BENCHMARK_PID}" &
    MONITOR_PID=$!
    set +e
    wait "${BENCHMARK_PID}"; BENCHMARK_STATUS=$?
    wait "${MONITOR_PID}"
    set -e
    echo "BENCHMARK_EXIT=${BENCHMARK_STATUS}"
    exit "${BENCHMARK_STATUS}"
) 2>&1 | tee "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}
if [[ "${STATUS}" -eq 0 ]]; then
    : >"${RESULT_DIR}/BENCHMARK_EXIT_0"
fi
echo "EXIT=${STATUS}"
echo "LOG=${LOG_PATH}"
echo "RESULT=${OUTPUT_PATH}"
echo "GPU_PROCESS_LOG=${GPU_PROCESS_LOG}"
exit "${STATUS}"
