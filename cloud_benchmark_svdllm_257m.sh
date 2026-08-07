#!/usr/bin/env bash

set -u

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/workspace-SR006.nfs2/dimativator/spectral-wd-257m}
RESULT_DIR=${RESULT_DIR:-/home/jovyan/results/svdllm-inference-257m}
LOG_DIR=${LOG_DIR:-/home/jovyan/logs}
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/benchmark-svdllm-257m-$(date +%F_%H%M%S).log"
OUTPUT_PATH="${RESULT_DIR}/results.json"
shopt -s nullglob
CHECKPOINTS=("${CHECKPOINT_ROOT}"/llama257m_*/ckpts/latest/main.pt)
if [[ "${#CHECKPOINTS[@]}" -ne 17 ]]; then
    echo "Expected 17 staged checkpoints, found ${#CHECKPOINTS[@]}" >&2
    exit 2
fi

set -o pipefail
PYTHONUNBUFFERED=1 python src/compression/benchmark_svd_llm_inference.py \
    "${CHECKPOINTS[@]}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --calib_batches 16 \
    --calib_batch_size 8 \
    --warmup_steps 10 \
    --timed_steps 50 \
    --calibration_tokens "${CHECKPOINT_ROOT}/calibration/val.bin" \
    --output "${OUTPUT_PATH}" 2>&1 | tee "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}

if [ -f "${OUTPUT_PATH}" ]; then
    python - "${OUTPUT_PATH}" "${HF_REPO_ID:-DimaTivator/spectral-wd-257m-checkpoints}" \
        >>"${LOG_PATH}" 2>&1 <<'PY'
import sys

from huggingface_hub import HfApi

result_path, repo_id = sys.argv[1:]
HfApi().upload_file(
    path_or_fileobj=result_path,
    path_in_repo="results/cloud-h100-inference-speed.json",
    repo_id=repo_id,
    repo_type="dataset",
)
print("RESULT_UPLOADED=results/cloud-h100-inference-speed.json", flush=True)
PY
    UPLOAD_STATUS=$?
    if [ "${STATUS}" -eq 0 ] && [ "${UPLOAD_STATUS}" -ne 0 ]; then
        STATUS=${UPLOAD_STATUS}
    fi
fi

echo "EXIT=${STATUS}"
echo "LOG=${LOG_PATH}"
echo "RESULT=${OUTPUT_PATH}"
tail -200 "${LOG_PATH}"
exit 0
