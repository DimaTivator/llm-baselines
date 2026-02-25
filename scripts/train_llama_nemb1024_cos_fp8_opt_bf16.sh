#!/usr/bin/env bash
set -euo pipefail

NGPUS=${1:-1}
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}

# ─── Model ──────────────────────────────
N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256

# ─── Training ───────────────────────────
ITERATIONS=80000
WARMUP=8000
BATCH_SIZE=16
ACC_STEPS=4
LR=3e-4
WEIGHT_DECAY=0.1

# ─── Launch ───────────────────────────────────────────────────────────────────
torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
    --distributed-backend nccl \
    \
    --dataset slimpajama \
    --datasets-dir "${DATASETS_DIR}" \
    --sequence-length ${SEQ_LEN} \
    \
    --model fp8_llama \
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
    --beta2 0.95 \
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
    --fp8-fabit E4M3 \
    --fp8-fwbit E4M3 \
    --fp8-babit E5M2 \
    --fp8-bwbit E5M2 \
    --fp8-group-size 16 \
    \
    --eval-interval 500 \
    --eval-batches 32 \
    --log-interval 50 \
    --latest-ckpt-interval 5000 \
    \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}"
