#!/usr/bin/env bash

set -euo pipefail

GPU_INDEX="${GPU_INDEX:-7}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
PYTHON_BIN="${PYTHON_BIN:-/data/users/dimativator/anaconda3/envs/eff-pretrain/bin/python}"
EXP_ROOT="${EXP_ROOT:-exps/llama500m_checkpoints}"
EXPERIMENT_LIST="${EXPERIMENT_LIST:-src/compression/svdllm_500m_speed_checkpoints.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exps/svdllm_m16_margin_quality_500m_20260821}"
LOG_PATH="${LOG_PATH:-${OUTPUT_ROOT}/runner.log}"

free_mib="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( free_mib < MIN_FREE_MIB )); then
    echo "GPU ${GPU_INDEX} has ${free_mib} MiB free; need ${MIN_FREE_MIB} MiB"
    exit 1
fi

mapfile -t experiments < <(sed '/^[[:space:]]*$/d' "${EXPERIMENT_LIST}")
if [[ "${#experiments[@]}" -ne 6 ]]; then
    echo "Expected six checkpoints, found ${#experiments[@]}"
    exit 1
fi

checkpoints=()
for experiment in "${experiments[@]}"; do
    checkpoint="${EXP_ROOT}/${experiment}/ckpts/latest/main.pt"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing checkpoint: ${checkpoint}"
        exit 1
    fi
    checkpoints+=("${checkpoint}")
done

mkdir -p "${OUTPUT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export DATASETS_DIR="${DATASETS_DIR:-/data/datasets/fineweb-edu-100BT/sample/100BT}"
export EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-/data/users/dimativator/evals_cache}"
export PYTHONPATH="${PYTHONPATH:-./src}"

{
    date -u +"START=%Y-%m-%dT%H:%M:%SZ"
    nvidia-smi --id="${GPU_INDEX}" --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m compression.svdllm_aligned_margin_quality_sweep \
        --checkpoints "${checkpoints[@]}" \
        --output_root "${OUTPUT_ROOT}" \
        --device cuda:0 \
        --auto_rank_multiple 16 \
        --margins -32 -16 0 16 32 \
        --eval_batches 64 \
        --eval_batch_size 8 \
        --calib_batches 16
    status=$?
    nvidia-smi --id="${GPU_INDEX}" --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
    echo "EXIT=${status}"
    exit "${status}"
} 2>&1 | tee -a "${LOG_PATH}"
