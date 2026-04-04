#!/usr/bin/env bash
set -euo pipefail

DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
RESULTS_DIR=${RESULTS_DIR:-"./exps/tuning/baseline_lr"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}
MAX_PARALLEL=${MAX_PARALLEL:-4}

# Parse GPU list from CUDA_VISIBLE_DEVICES, or default to 0..MAX_PARALLEL-1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    unset CUDA_VISIBLE_DEVICES
else
    GPUS=()
    for ((i=0; i<MAX_PARALLEL; i++)); do GPUS+=("$i"); done
fi

# Never run more jobs than GPUs
(( ${#GPUS[@]} < MAX_PARALLEL )) && MAX_PARALLEL=${#GPUS[@]}

# ─── Model ────────────────────────────────────────────────────────────────────
N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256

# ─── Training ─────────────────────────────────────────────────────────────────
ITERATIONS=39250
WARMUP=3925
BATCH_SIZE=32
ACC_STEPS=4
WEIGHT_DECAY=0.1

# ─── Sweep ────────────────────────────────────────────────────────────────────
LRS=(1e-4 5e-4 1e-3 2e-3)

PIDS=()

cleanup() {
    trap '' INT TERM EXIT
    echo ""
    echo "=== Stopping all runs ==="
    for pid in "${PIDS[@]}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

GPU_IDX=0

for LR in "${LRS[@]}"; do

    # Wait for a free slot when all are busy
    if (( ${#PIDS[@]} >= MAX_PARALLEL )); then
        wait -n -p DONE_PID "${PIDS[@]}"
        PIDS=("${PIDS[@]/$DONE_PID/}")
        local_pids=()
        for p in "${PIDS[@]}"; do [[ -n "$p" ]] && local_pids+=("$p"); done
        PIDS=("${local_pids[@]}")
    fi

    GPU=${GPUS[$((GPU_IDX % ${#GPUS[@]}))]}

    echo "=== Launching: lr=${LR} on GPU ${GPU} ==="

    CUDA_VISIBLE_DEVICES=${GPU} torchrun \
        --standalone --nproc_per_node=1 \
        --master_port=$((29500 + GPU_IDX)) \
        src/main.py \
        --distributed-backend nccl \
        --experiment-name "adamw_lr${LR}_wd${WEIGHT_DECAY}" \
        \
        --dataset fineweb \
        --datasets-dir "${DATASETS_DIR}" \
        --sequence-length ${SEQ_LEN} \
        --streaming \
        --workers 8 \
        \
        --model llama \
        --n-layer ${N_LAYER} \
        --n-embd  ${N_EMBD} \
        --n-head  ${N_HEAD} \
        --multiple-of ${MULTIPLE_OF} \
        --dtype bfloat16 \
        \
        --opt adamw \
        --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --beta1 0.9 \
        --beta2 0.99 \
        --grad-clip 1.0 \
        \
        --scheduler cos \
        --warmup-steps ${WARMUP} \
        --iterations ${ITERATIONS} \
        \
        --batch-size ${BATCH_SIZE} \
        --acc-steps ${ACC_STEPS} \
        \
        --eval-interval 500 \
        --eval-batches 32 \
        --downstream-eval-enabled \
        --downstream-eval-interval 2000 \
        --downstream-task-group basic_v2 \
        --lm-eval-enabled \
        --lm-eval-interval 2000 \
        --lm-eval-datasets wikitext103 \
        --log-interval 50 \
        --latest-ckpt-interval 0 \
        \
        --results-base-folder "${RESULTS_DIR}" \
        --wandb \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-group adamw-lr-tuning \
        --wandb-tags baseline tuning &

    PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))

done

echo "=== Waiting for all ${#PIDS[@]} runs to finish ==="
wait
echo "=== All sweeps done ==="
