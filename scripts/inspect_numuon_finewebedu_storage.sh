#!/bin/bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-"/workspace-SR006.nfs2/dimativator/finewebedu_h200"}
CACHE_ROOT=${CACHE_ROOT:-"/workspace-SR006.nfs2/dimativator/hf-cache-finewebedu-h200"}

df -h /workspace-SR006.nfs2
for path in "$CACHE_ROOT" "$DATA_ROOT/tokenized/train.bin" "$DATA_ROOT/tokenized/val.bin"; do
    if [ -e "$path" ]; then
        stat -c 'PATH=%n TYPE=%F SIZE=%s' "$path"
    else
        echo "PATH=$path MISSING"
    fi
done
