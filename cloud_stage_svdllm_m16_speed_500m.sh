#!/usr/bin/env bash

set -u

HF_REPO_ID=${HF_REPO_ID:-DimaTivator/spectral-wd-500m-checkpoints}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/workspace-SR006.nfs2/dimativator/spectral-wd-500m}
EXPERIMENT_LIST=${EXPERIMENT_LIST:-src/compression/svdllm_500m_speed_checkpoints.txt}
LOG_DIR=${LOG_DIR:-/home/jovyan/logs}
mkdir -p "${LOG_DIR}" "${CHECKPOINT_ROOT}"
LOG_PATH="${LOG_DIR}/stage-svdllm-m16-speed-500m-$(date +%F_%H%M%S).log"
EXTRA_ARGS=()
if [[ "${INSPECT_ONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--inspect_only)
fi

set -o pipefail
python src/compression/stage_svdllm_selected_cloud.py \
    --repo_id "${HF_REPO_ID}" \
    --destination "${CHECKPOINT_ROOT}" \
    --experiment_list "${EXPERIMENT_LIST}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}

echo "EXIT=${STATUS}"
echo "LOG=${LOG_PATH}"
tail -100 "${LOG_PATH}"
exit "${STATUS}"
