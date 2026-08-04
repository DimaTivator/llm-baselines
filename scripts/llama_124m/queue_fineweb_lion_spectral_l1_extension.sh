#!/bin/bash
set -euo pipefail

CURRENT_SESSION=${CURRENT_SESSION:-"lion124m_sl1_lr1e3_gpu1_20260803"}
REPO_DIR=${REPO_DIR:-"$(pwd)"}
CURRENT_LOG=${CURRENT_LOG:-"${REPO_DIR}/logs/${CURRENT_SESSION}.log"}
POLL_SECONDS=${POLL_SECONDS:-60}
SPECTRAL_L1_COEFS=${SPECTRAL_L1_COEFS:-"1.5 2 2.5 3"}

echo "Waiting for tmux session ${CURRENT_SESSION}"
while tmux has-session -t "${CURRENT_SESSION}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
done

# The current queue prints this marker only after its launcher returns. Do not
# consume more GPU time if any of its runs failed or was interrupted.
if ! grep -aEq 'SWEEP_EXIT=0\r?$' "${CURRENT_LOG}"; then
    echo "Current sweep did not finish successfully; extension will not start."
    exit 1
fi

echo "Current sweep completed; starting cf=${SPECTRAL_L1_COEFS}"
export SPECTRAL_L1_COEFS
exec bash "${REPO_DIR}/scripts/llama_124m/train_fineweb_lion_spectral_l1_sweep.sh"
