#!/usr/bin/env bash
set -euo pipefail

NGPUS=${1:-1}
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}

# --- Model ---
N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256

# --- Training ---
ITERATIONS=39250
WARMUP=3925
BATCH_SIZE=32
ACC_STEPS=4
LR=5e-4
WEIGHT_DECAY=0.1

# --- Launch ---
torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
    --distributed-backend nccl \
    --experiment-name "coat_adamw_fp8_fineweb_lr5e-4_wd0.1" \
    \
    --dataset fineweb \
    --datasets-dir "${DATASETS_DIR}" \
    --sequence-length ${SEQ_LEN} \
    --streaming \
    --workers 8 \
    \
    --model llama \
    --n-layer ${N_LAYER} \
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --multiple-of ${MULTIPLE_OF} \
    --dtype bfloat16 \
    \
    --opt coat_adamw \
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
    --fp8 \
    --fp8-optim \
    --fp8-fabit E4M3 \
    --fp8-fwbit E4M3 \
    --fp8-babit E5M2 \
    --fp8-bwbit E5M2 \
    --fp8-group-size 16 \
    --fp8-qgroup-size 128 \
    --fp8-first-order-bit E4M3 \
    --fp8-second-order-bit E4M3 \
    --fp8-expansion expand \
    \
    --eval-interval 500 \
    --eval-batches 32 \
    --log-interval 50 \
    --latest-ckpt-interval 5000 \
    \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-tags fineweb fp8 coat
