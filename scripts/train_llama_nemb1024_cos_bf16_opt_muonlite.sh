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
LR=7.5e-4            # Muon-optimal LR (3e-3 from LITE Table 2, scaled by √(65536/1048576))
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
    --opt muonlite \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --beta1 0.9 \
    --beta2 0.95 \
    --grad-clip 1.0 \
    \
    --lite-beta1 -0.25 \
    --lite-beta2 1.0 \
    --lite-chi 2.0 \
    --lite-chi-adamw 4.0 \
    --lite-subspace-ratio 0.1 \
    --lite-ns-steps 6 \
    --lite-muon-theta 0.95 \
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
