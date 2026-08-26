#!/bin/bash
set -euo pipefail

CACHE_ROOT=${CACHE_ROOT:-"/workspace-SR006.nfs2/dimativator/hf-cache-finewebedu-h200"}
DATA_ROOT=${DATA_ROOT:-"/workspace-SR006.nfs2/dimativator/finewebedu_h200"}

if [ "$CACHE_ROOT" != "/workspace-SR006.nfs2/dimativator/hf-cache-finewebedu-h200" ]; then
    echo "Refusing unexpected CACHE_ROOT=$CACHE_ROOT"
    exit 2
fi

echo "CACHE_CLEANUP_START=$CACHE_ROOT"
rm -rf -- "$CACHE_ROOT"

for path in "$DATA_ROOT/tokenized/train.bin" "$DATA_ROOT/tokenized/val.bin"; do
    if [ -e "$path" ]; then
        echo "Removing incomplete output $path"
        rm -f -- "$path"
    fi
done

echo "CACHE_CLEANUP=OK"
df -h /workspace-SR006.nfs2
