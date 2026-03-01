#!/usr/bin/env bash
set -euo pipefail

NGPUS=${1:-1}
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}

# ─── Model ────────────────────────────────────────────────────────────────────
N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256

# ─── Training ─────────────────────────────────────────────────────────────────
# Chinchilla-optimal for 1 GPU (5.142B tokens, 65536 tokens/step)
ITERATIONS=80000
WARMUP=8000          # 10% of iterations
BATCH_SIZE=16
ACC_STEPS=4
LR=3e-4
WEIGHT_DECAY=0.1

# ─── SOLO config (4/2-bit as recommended in paper) ───────────────────────────
# 1st state: 4-bit signed (dynamic exponent), 2nd state: 2-bit unsigned (qema)
# Beta1 reduced to 0.3 for training from scratch per paper Section 3.4
SOLO_BITS="4 2"
SOLO_QUANTIZERS="de qema"
SOLO_BLOCK_SIZES="128 128"
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
    --opt solo_adamw \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --beta1 ${BETA1} \
    --beta2 0.95 \
    --grad-clip 1.0 \
    \
    --solo-bits ${SOLO_BITS} \
    --solo-quantizers ${SOLO_QUANTIZERS} \
    --solo-block-sizes ${SOLO_BLOCK_SIZES} \
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
