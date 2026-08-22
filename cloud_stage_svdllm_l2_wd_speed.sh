#!/usr/bin/env bash

set -euo pipefail

HF_REPO_ID="${HF_REPO_ID:-DimaTivator/svdllm-l2-wd-speed-checkpoints}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace-SR006.nfs2/dimativator/svdllm-l2-wd-speed-20260822}"
EXPERIMENT_LIST="${EXPERIMENT_LIST:-src/compression/svdllm_l2_wd_speed_checkpoints.txt}"
LOG_DIR="${LOG_DIR:-/home/jovyan/logs}"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_ROOT}"
LOG_PATH="${LOG_DIR}/stage-svdllm-l2-wd-speed-$(date +%F_%H%M%S).log"

set -o pipefail
python src/compression/stage_svdllm_selected_cloud.py \
    --repo_id "${HF_REPO_ID}" \
    --destination "${CHECKPOINT_ROOT}" \
    --experiment_list "${EXPERIMENT_LIST}" 2>&1 | tee "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}
echo "STAGE_EXIT=${STATUS}"
echo "LOG=${LOG_PATH}"
exit "${STATUS}"
