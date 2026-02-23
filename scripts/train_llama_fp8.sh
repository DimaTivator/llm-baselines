#!/usr/bin/env bash
# train_llama_fp8.sh  —  FP8 run for the BF16 vs FP8 comparison.
#
# Architecture: Llama 124M (12L / 768D / 12H) — identical to train_llama_bf16.sh.
# FP8 coverage (COAT):
#   - Activations   : fp8_rmsnorm, fp8_silu/mul, fp8_linear (all projections)
#   - Optimizer     : CoatAdamW — first- and second-order moments in FP8
#
# Usage:
#   bash scripts/train_llama_fp8.sh [NUM_GPUS]

set -euo pipefail

NGPUS=${1:-1}
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}

# ─── Model — MUST match the BF16 script exactly ──────────────────────────────
N_LAYER=12
N_EMBD=768
N_HEAD=12
SEQ_LEN=1024
MULTIPLE_OF=256

# ─── Training — MUST match the BF16 script exactly ───────────────────────────
ITERATIONS=50000
WARMUP=1000
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
    --opt coat_adamw \
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
    --latest-ckpt-interval 10000 \
    \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}"
