#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=$(cat ~/.radfan_wandb_api_key)
export HF_TOKEN=$(cat ~/.hf_token)

NGPUS=1
LOCAL_DATA_PATH=$(eval echo ~/terra_projects/optimizers/LoRA/c4_val)
RESULTS_DIR=${RESULTS_DIR:-"./exps"}  # This is relative to script dir too
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain-TEST"}

# ─── Model ────────────────────────────────────────────────────────────────────
N_LAYER=5
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256

# ─── Training ─────────────────────────────────────────────────────────────────
ITERATIONS=1000  # Small for debug
WARMUP=100
BATCH_SIZE=4
ACC_STEPS=2
LR=3e-4
WEIGHT_DECAY=0.1

# ─── Launch ───────────────────────────────────────────────────────────────────
torchrun --standalone --nproc_per_node="${NGPUS}" ./../src/main.py \
    --distributed-backend nccl \
    \
    --dataset c4 \
    --local_data \
    --local_data_path "${LOCAL_DATA_PATH}" \
    --sequence-length ${SEQ_LEN} \
    --data-in-ram \
    \
    --model llama \
    --n-layer ${N_LAYER} \
    --n-embd  ${N_EMBD} \
    --n-head  ${N_HEAD} \
    --multiple-of ${MULTIPLE_OF} \
    --dtype float32 \
    \
    --opt swan \
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
    --eval-interval 100 \
    --eval-batches 8 \
    --log-interval 10 \
    --latest-ckpt-interval 500 \
    \
    --results-base-folder "${RESULTS_DIR}" \
    --no-local-save \
    --wandb \
    --wandb-project "${WANDB_PROJECT}"