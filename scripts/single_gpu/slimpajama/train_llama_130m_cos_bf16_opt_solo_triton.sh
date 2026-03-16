#!/usr/bin/env bash
set -euo pipefail

NGPUS=${1:-1}
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}

# ─── Model (Llama-130M: 12L/768D/12H) ────────────────────────────────────────
N_LAYER=12
N_EMBD=768
N_HEAD=12
SEQ_LEN=1024
MULTIPLE_OF=256

# ─── Training ─────────────────────────────────────────────────────────────────
# Chinchilla-optimal for 1 GPU (~3.277B tokens, 65536 tokens/step)
ITERATIONS=50000
WARMUP=5000          # 10% of iterations
BATCH_SIZE=16
ACC_STEPS=4
LR=3e-4
WEIGHT_DECAY=0.1

# ─── SOLO Triton config (4/2-bit, fused kernel) ──────────────────────────────
# Hardcoded: 1st state 4-bit DE signed, 2nd state 2-bit qema unsigned
# Beta1 reduced to 0.3 for training from scratch per paper Section 3.4
SOLO_BLOCK_SIZE=128
SOLO_QUANTILE=0.1
BETA1=0.3

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
    --opt solo_triton_adamw \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --beta1 ${BETA1} \
    --beta2 0.95 \
    --grad-clip 1.0 \
    \
    --solo-block-sizes ${SOLO_BLOCK_SIZE} ${SOLO_BLOCK_SIZE} \
    --solo-quantile ${SOLO_QUANTILE} \
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
