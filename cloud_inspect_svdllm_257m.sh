#!/usr/bin/env bash

set -u

ROOT=${CHECKPOINT_ROOT:-/workspace-SR006.nfs2/dimativator/spectral-wd-257m}
COUNT=0
BYTES=0

shopt -s nullglob
for CHECKPOINT in "${ROOT}"/llama257m_*/ckpts/latest/main.pt; do
    COUNT=$((COUNT + 1))
    SIZE=$(stat -c %s "${CHECKPOINT}")
    BYTES=$((BYTES + SIZE))
done

echo "STAGED_CHECKPOINTS=${COUNT}"
echo "STAGED_CHECKPOINT_BYTES=${BYTES}"
df -B1 "${ROOT}"
