#!/bin/bash
set -euo pipefail

LOG_ROOT=${LOG_ROOT:-"/home/jovyan/logs/adamw_l2_svdllm"}
RESULT_ROOT=${RESULT_ROOT:-"/home/jovyan/results/adamw_l2_svdllm"}

echo "=== recent rank logs ==="
if [ -d "$LOG_ROOT" ]; then
    find "$LOG_ROOT" -maxdepth 1 -type f -name '*.log' \
        -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' \
        | sort -nr \
        | head -30 \
        | cut -d' ' -f2-
else
    echo "missing: $LOG_ROOT"
fi

echo "=== tail of each recent rank log ==="
if [ -d "$LOG_ROOT" ]; then
    while IFS= read -r log_file; do
        echo "--- $log_file ---"
        tail -40 "$log_file"
    done < <(
        find "$LOG_ROOT" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' \
            | sort -nr \
            | head -20 \
            | cut -d' ' -f2-
    )
fi

echo "=== persistent result files ==="
if [ -d "$RESULT_ROOT" ]; then
    find "$RESULT_ROOT" -type f \
        \( -name 'svdllm_margin_table.json' \
        -o -name 'svdllm_margin_table.md' \
        -o -name 'COMPLETE' \) \
        -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' \
        | sort
else
    echo "missing: $RESULT_ROOT"
fi
