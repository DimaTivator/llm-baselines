#!/usr/bin/env bash
# train_llama_cos_bf16.sh  —  BF16 baseline with cosine LR schedule.
#
# Architecture: Llama 124M (12L / 768D / 12H), same as train_llama_bf16.sh.
# Differs from train_llama_bf16.sh only in the LR schedule:
#   - scheduler : cosine  (vs WSD in the reference script)
#   - warmup    : 10% of iterations (2000 steps)
#
# Usage:
#   bash scripts/train_llama_cos_bf16.sh [NUM_GPUS]

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
MULTIPLE_OF=256

# ─── Training ─────────────────────────────────────────────────────────────────
# Chinchilla-optimal for 1 GPU (3.242B tokens, 65536 tokens/step)
ITERATIONS=50000
WARMUP=5000          # 10% of total iterations
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
    --scheduler cos \
    --warmup-steps ${WARMUP} \
    --iterations ${ITERATIONS} \
    \
    --batch-size ${BATCH_SIZE} \
    --acc-steps ${ACC_STEPS} \
    \
    --eval-interval 500 \
    --eval-batches 32 \
    --log-interval 50 \
    --latest-ckpt-interval 5000 \
    \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}"
