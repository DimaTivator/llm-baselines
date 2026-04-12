#!/usr/bin/env bash
set -euo pipefail

NGPUS=${1:-1}
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}

N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256

ITERATIONS=78500
WARMUP=7850
BATCH_SIZE=32
ACC_STEPS=4
LR=1e-3
WEIGHT_DECAY=0.1

BETA1=0.9
BETA2=0.999
BETA3=0.9999
ALPHA=8
BETA3_WARMUP=${ITERATIONS}
ALPHA_WARMUP=${ITERATIONS}

torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
    --distributed-backend nccl \
    --experiment-name "ademamix_lr1e-3_wd0.1_2xC" \
    \
    --dataset fineweb \
    --datasets-dir "${DATASETS_DIR}" \
    --sequence-length ${SEQ_LEN} \
    --streaming \
    --workers 8 \
    \
    --model llama \
    --n-layer ${N_LAYER} \
    --n-embd  ${N_EMBD} \
    --n-head  ${N_HEAD} \
    --multiple-of ${MULTIPLE_OF} \
    --dtype bfloat16 \
    \
    --opt ademamix \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --beta1 ${BETA1} \
    --beta2 ${BETA2} \
    --ademamix_beta3 ${BETA3} \
    --ademamix_alpha ${ALPHA} \
    --ademamix_beta3_warmup_steps ${BETA3_WARMUP} \
    --ademamix_alpha_warmup_steps ${ALPHA_WARMUP} \
    --grad-clip 0.5 \
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
    --downstream-eval-enabled \
    --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled \
    --lm-eval-interval 2000 \
    --lm-eval-datasets wikitext103 \
    --log-interval 50 \
    --latest-ckpt-interval 5000 \
    \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group 2xChinchilla \
    --wandb-tags baseline bf16 ademamix
