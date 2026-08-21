#!/usr/bin/env bash

set -u

HF_REPO_ID=${HF_REPO_ID:-DimaTivator/spectral-wd-500m-checkpoints}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/workspace-SR006.nfs2/dimativator/spectral-wd-500m}
EXPERIMENT_LIST=${EXPERIMENT_LIST:-src/compression/svdllm_500m_speed_checkpoints.txt}
RESULT_DIR=${RESULT_DIR:-/workspace-SR006.nfs2/dimativator/results/svdllm-m16-compile-b256-500m}
LOG_DIR=${LOG_DIR:-/home/jovyan/logs}
COMPILE_MODE=${COMPILE_MODE:-max-autotune}
UPLOAD_RESULTS=${UPLOAD_RESULTS:-1}
CHECK_ONLY=${CHECK_ONLY:-0}
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

OUTPUT_PATH="${RESULT_DIR}/results.json"
GPU_PROCESS_LOG="${RESULT_DIR}/gpu-processes.csv"
LOG_PATH="${LOG_DIR}/benchmark-svdllm-m16-compile-b256-500m-$(date +%F_%H%M%S).log"
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-svdllm-m16-b256-500m}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-svdllm-m16-b256-500m}

mapfile -t EXPERIMENTS < <(sed '/^[[:space:]]*$/d' "${EXPERIMENT_LIST}")
if [[ "${#EXPERIMENTS[@]}" -ne 6 ]]; then
    echo "Expected six experiments, found ${#EXPERIMENTS[@]}"
    exit 0
fi

CHECKPOINTS=()
for experiment in "${EXPERIMENTS[@]}"; do
    checkpoint="${CHECKPOINT_ROOT}/${experiment}/ckpts/latest/main.pt"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing checkpoint: ${checkpoint}"
        exit 0
    fi
    CHECKPOINTS+=("${checkpoint}")
done
CALIBRATION="${CHECKPOINT_ROOT}/calibration/val.bin"
if [[ ! -f "${CALIBRATION}" ]]; then
    echo "Missing calibration tokens: ${CALIBRATION}"
    exit 0
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
    python -m py_compile src/compression/benchmark_svd_llm_inference.py src/compression/stage_svdllm_selected_cloud.py
    echo "CHECK_ONLY_OK=1"
    echo "CHECKPOINTS=${#CHECKPOINTS[@]}"
    exit 0
fi

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
    set -e
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    if [[ "${GPU_NAME}" != *H100* ]]; then
        echo "Expected H100, found ${GPU_NAME}"
        exit 1
    fi
    INITIAL_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)
    if [[ "${INITIAL_PROCESSES}" -ne 0 ]]; then
        echo "GPU is not isolated before benchmark: ${INITIAL_PROCESSES} compute processes"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
        exit 1
    fi
    nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,memory.free --format=csv

    : >"${GPU_PROCESS_LOG}"
    set +e
    PYTHONUNBUFFERED=1 PYTHONPATH=src python src/compression/benchmark_svd_llm_inference.py \
        "${CHECKPOINTS[@]}" \
        --device cuda:0 \
        --dtype bfloat16 \
        --compile_mode "${COMPILE_MODE}" \
        --disable_inductor_pattern_matcher \
        --auto_rank_multiple 16 \
        --batch_sizes 256 \
        --calib_batches 16 \
        --calib_batch_size 8 \
        --warmup_steps 10 \
        --timed_steps 50 \
        --calibration_tokens "${CALIBRATION}" \
        --output "${OUTPUT_PATH}" &
    BENCHMARK_PID=$!
    monitor_gpu_processes "${BENCHMARK_PID}" &
    MONITOR_PID=$!
    wait "${BENCHMARK_PID}"
    BENCHMARK_STATUS=$?
    wait "${MONITOR_PID}"
    set -e

    nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,memory.free --format=csv
    echo "BENCHMARK_EXIT=${BENCHMARK_STATUS}"
    exit "${BENCHMARK_STATUS}"
) 2>&1 | tee "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}

if [[ -f "${OUTPUT_PATH}" && "${UPLOAD_RESULTS}" == "1" ]]; then
    python - "${OUTPUT_PATH}" "${GPU_PROCESS_LOG}" "${HF_REPO_ID}" >>"${LOG_PATH}" 2>&1 <<'PY'
import sys

from huggingface_hub import HfApi

result_path, process_path, repo_id = sys.argv[1:]
api = HfApi()
for local_path, remote_path in (
    (result_path, "results/cloud-h100-m16-compile-b256-500m.json"),
    (process_path, "results/cloud-h100-m16-compile-b256-500m-gpu-processes.csv"),
):
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"RESULT_UPLOADED={remote_path}", flush=True)
PY
    UPLOAD_STATUS=$?
    if [[ "${STATUS}" -eq 0 && "${UPLOAD_STATUS}" -ne 0 ]]; then
        STATUS=${UPLOAD_STATUS}
    fi
fi

echo "EXIT=${STATUS}"
echo "LOG=${LOG_PATH}"
echo "RESULT=${OUTPUT_PATH}"
echo "GPU_PROCESS_LOG=${GPU_PROCESS_LOG}"
tail -200 "${LOG_PATH}"
exit 0
