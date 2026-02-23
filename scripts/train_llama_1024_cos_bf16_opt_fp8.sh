#!/usr/bin/env bash
# train_llama_1024_cos_bf16_opt_fp8.sh  —  BF16 activations + FP8 optimizer, 1024-wide, cosine LR.
#
# Architecture: 12L / 1024D / 8H  (head_dim=128, same as Quartet-II 100M)
#   multiple_of=256 → hidden_dim = 2816
# Forward/backward pass runs in BF16 (--model llama, no --fp8).
# Optimizer moments stored in FP8 via CoatAdamW.
# LR schedule: cosine with 10% warmup (Chinchilla-optimal for 1 GPU: 5.142B tokens).
#
# Usage:
#   bash scripts/train_llama_1024_cos_bf16_opt_fp8.sh [NUM_GPUS]

set -euo pipefail

NGPUS=${1:-1}
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}

# ─── Model (12L / 1024D / 8H) ────────────────────────────────────────────────
N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256   # hidden_dim = 2816

# ─── Training ─────────────────────────────────────────────────────────────────
# Chinchilla-optimal for 1 GPU (5.142B tokens, 65536 tokens/step)
ITERATIONS=80000
WARMUP=8000          # 10% of total iterations
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
    --opt coat_adamw \
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
    --fp8-optim \
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
    --wandb-project "${WANDB_PROJECT}"
