#!/usr/bin/env bash

set -u

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/workspace-SR006.nfs2/dimativator/spectral-wd-257m}
RESULT_DIR=${RESULT_DIR:-/home/jovyan/results/svdllm-inference-257m}
LOG_DIR=${LOG_DIR:-/home/jovyan/logs}
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/benchmark-svdllm-257m-$(date +%F_%H%M%S).log"
OUTPUT_PATH="${RESULT_DIR}/results.json"

PYTHONUNBUFFERED=1 python src/compression/benchmark_svd_llm_inference.py \
    "${CHECKPOINT_ROOT}"/llama257m_*/ckpts/latest/main.pt \
    --device cuda:0 \
    --dtype bfloat16 \
    --calib_batches 16 \
    --calib_batch_size 8 \
    --warmup_steps 10 \
    --timed_steps 50 \
    --calibration_tokens "${CHECKPOINT_ROOT}/calibration/val.bin" \
    --output "${OUTPUT_PATH}" >"${LOG_PATH}" 2>&1
STATUS=$?

echo "EXIT=${STATUS}"
echo "LOG=${LOG_PATH}"
echo "RESULT=${OUTPUT_PATH}"
tail -200 "${LOG_PATH}"
exit 0
