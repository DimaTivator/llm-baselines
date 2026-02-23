#!/usr/bin/env bash
# train_llama_bf16.sh  —  BF16 baseline for the BF16 vs FP8 comparison.
#
# Architecture: Llama 124M (12L / 768D / 12H), same size as GPT-2 124M.
# Use alongside train_llama_fp8.sh for an apples-to-apples BF16 vs FP8 run.
#
# Usage:
#   bash scripts/train_llama_bf16.sh [NUM_GPUS]

set -euo pipefail

NGPUS=${1:-1}
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}

# ─── Model (Llama 124M) ───────────────────────────────────────────────────────
N_LAYER=12
N_EMBD=768
N_HEAD=12
SEQ_LEN=1024
MULTIPLE_OF=256   # MLP hidden dim rounding

# ─── Training ─────────────────────────────────────────────────────────────────
ITERATIONS=50000
WARMUP=1000
BATCH_SIZE=16
ACC_STEPS=4          # effective batch = NGPUS * 16 * 4 * 1024 tokens
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
    --beta2 0.95 \
    --grad-clip 1.0 \
    \
    --scheduler wsd \
    --warmup-steps ${WARMUP} \
    --iterations ${ITERATIONS} \
    --wsd-fract-decay 0.1 \
    --decay-type cosine \
    \
    --batch-size ${BATCH_SIZE} \
    --acc-steps ${ACC_STEPS} \
    \
    --eval-interval 500 \
    --eval-batches 32 \
    --log-interval 50 \
    --latest-ckpt-interval 10000 \
    \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}"
