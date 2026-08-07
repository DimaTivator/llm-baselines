#!/usr/bin/env bash

set -u

REPO_ID=${HF_REPO_ID:-DimaTivator/spectral-wd-257m-checkpoints}
DESTINATION=${CHECKPOINT_ROOT:-/workspace-SR006.nfs2/dimativator/spectral-wd-257m}
LOG_DIR=${LOG_DIR:-/home/jovyan/logs}
mkdir -p "${LOG_DIR}" "${DESTINATION}"
LOG_PATH="${LOG_DIR}/stage-svdllm-257m-$(date +%F_%H%M%S).log"

python src/compression/stage_svdllm_257m_cloud.py \
    --repo_id "${REPO_ID}" \
    --destination "${DESTINATION}" >"${LOG_PATH}" 2>&1
STATUS=$?

echo "EXIT=${STATUS}"
echo "LOG=${LOG_PATH}"
tail -100 "${LOG_PATH}"
exit 0
