#!/usr/bin/env bash

set -u

EXPERIMENT=llama257m_adamw-spectral-l1-reg_wd1e-1_lr3e-3_sl1_1_finewebedu_erank
HF_REPO_ID=${HF_REPO_ID:-DimaTivator/spectral-wd-257m-checkpoints}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/workspace-SR006.nfs2/dimativator/spectral-wd-257m-compile-cf1}
RESULT_DIR=${RESULT_DIR:-/home/jovyan/results/svdllm-inference-compile-cf1}
LOG_DIR=${LOG_DIR:-/home/jovyan/logs}
COMPILE_MODE=${COMPILE_MODE:-max-autotune}
AUTO_RANK_MULTIPLE=${AUTO_RANK_MULTIPLE:-}
UPLOAD_RESULTS=${UPLOAD_RESULTS:-1}
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

LOG_PATH="${LOG_DIR}/benchmark-svdllm-compile-cf1-$(date +%F_%H%M%S).log"
RANK_SUFFIX=
RANK_ARGS=()
if [[ -n "${AUTO_RANK_MULTIPLE}" ]]; then
    RANK_SUFFIX="-auto-floor-m${AUTO_RANK_MULTIPLE}"
    RANK_ARGS=(--auto_rank_multiple "${AUTO_RANK_MULTIPLE}")
fi
OUTPUT_PATH="${RESULT_DIR}/results-${COMPILE_MODE}${RANK_SUFFIX}.json"
CHECKPOINT="${CHECKPOINT_ROOT}/${EXPERIMENT}/ckpts/latest/main.pt"
CALIBRATION="${CHECKPOINT_ROOT}/calibration/val.bin"
GPU_PROCESS_LOG="${RESULT_DIR}/gpu-processes-${COMPILE_MODE}${RANK_SUFFIX}.csv"

monitor_gpu_processes() {
    local benchmark_pid=$1
    while kill -0 "${benchmark_pid}" 2>/dev/null; do
        date -u +%Y-%m-%dT%H:%M:%SZ >>"${GPU_PROCESS_LOG}"
        nvidia-smi \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader >>"${GPU_PROCESS_LOG}"
        sleep 2
    done
}

set -o pipefail
(
    set -e
    python src/compression/stage_svdllm_257m_cloud.py \
        --repo_id "${HF_REPO_ID}" \
        --destination "${CHECKPOINT_ROOT}" \
        --expected_checkpoints 1 \
        --allow_patterns \
            "${EXPERIMENT}/ckpts/latest/main.pt" \
            "${EXPERIMENT}/summary.json" \
            "calibration/val.bin"

    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
        --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv

    : >"${GPU_PROCESS_LOG}"
    set +e
    PYTHONUNBUFFERED=1 python src/compression/benchmark_svd_llm_inference.py \
        "${CHECKPOINT}" \
        --device cuda:0 \
        --dtype bfloat16 \
        --compile_mode "${COMPILE_MODE}" \
        --disable_inductor_pattern_matcher \
        --calib_batches 16 \
        --calib_batch_size 8 \
        --warmup_steps 10 \
        --timed_steps 50 \
        --calibration_tokens "${CALIBRATION}" \
        "${RANK_ARGS[@]}" \
        --output "${OUTPUT_PATH}" &
    BENCHMARK_PID=$!
    monitor_gpu_processes "${BENCHMARK_PID}" &
    MONITOR_PID=$!
    wait "${BENCHMARK_PID}"
    BENCHMARK_STATUS=$?
    wait "${MONITOR_PID}"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
        --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv
    set -e
    exit "${BENCHMARK_STATUS}"
) 2>&1 | tee "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}

if [[ -f "${OUTPUT_PATH}" && "${UPLOAD_RESULTS}" == "1" ]]; then
    python - "${OUTPUT_PATH}" "${HF_REPO_ID}" "${COMPILE_MODE}" "${RANK_SUFFIX}" \
        >>"${LOG_PATH}" 2>&1 <<'PY'
import sys

from huggingface_hub import HfApi

result_path, repo_id, compile_mode, rank_suffix = sys.argv[1:]
path_in_repo = f"results/cloud-h100-compile-cf1-{compile_mode}{rank_suffix}.json"
HfApi().upload_file(
    path_or_fileobj=result_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="dataset",
)
print(f"RESULT_UPLOADED={path_in_repo}", flush=True)
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
