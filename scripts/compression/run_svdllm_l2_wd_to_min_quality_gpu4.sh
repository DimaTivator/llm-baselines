#!/usr/bin/env bash

set -euo pipefail

GPU_INDEX="${GPU_INDEX:-4}"
MIN_FREE_MIB="${MIN_FREE_MIB:-50000}"
PYTHON_BIN="${PYTHON_BIN:-/data/users/dimativator/anaconda3/envs/eff-pretrain/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exps/svdllm_l2_wd_to_min_quality_20260822}"
LOG_PATH="${LOG_PATH:-${OUTPUT_ROOT}/runner.log}"

free_mib="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( free_mib < MIN_FREE_MIB )); then
    echo "GPU ${GPU_INDEX} has ${free_mib} MiB free; need ${MIN_FREE_MIB} MiB"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export DATASETS_DIR="${DATASETS_DIR:-/data/datasets/fineweb-edu-100BT/sample/100BT}"
export EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-/data/users/dimativator/evals_cache}"
export PYTHONPATH="${PYTHONPATH:-./src}"

run_one() {
    local label=$1
    local model_size=$2
    local min_margin=$3
    local checkpoint=$4
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m compression.svdllm_to_min_sweep \
        --ckpt_path "${checkpoint}" \
        --output_dir "${OUTPUT_ROOT}/${label}" \
        --device cuda:0 \
        --model_size "${model_size}" \
        --checkpoint_label "${label}" \
        --min_margin "${min_margin}" \
        --eval_batches 64 \
        --eval_batch_size 8 \
        --calib_batches 16
}

{
    date -u +"START=%Y-%m-%dT%H:%M:%SZ"
    nvidia-smi --id="${GPU_INDEX}" --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
    run_one \
        "llama124m_sl1_0.01" "124M" -800 \
        "exps/cf_bruteforce_124M/llama124m_adamw-spectral-l1-reg_wd1e-1_lr5e-3_sl1_0.01_finewebedu_erank/ckpts/latest/main.pt"
    run_one \
        "llama257m_sl1_0.01" "257M" -1050 \
        "exps/cf_bruteforce_124M/llama257m_adamw-spectral-l1-reg_wd1e-1_lr3e-3_sl1_0.01_finewebedu_erank/ckpts/latest/main.pt"
    run_one \
        "llama500m_adamw" "500M" -1300 \
        "exps/llama500m_checkpoints/adamw_lr1e-3_finewebedu/main.pt"
    nvidia-smi --id="${GPU_INDEX}" --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
    echo "QUALITY_EXIT=0"
} 2>&1 | tee -a "${LOG_PATH}"
