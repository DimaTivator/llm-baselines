#!/usr/bin/env bash
set -euo pipefail

NGPUS=${1:-1}
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"./evals_cache"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}

N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256

ITERATIONS=39250
WARMUP=1963 # ~= 5%
BATCH_SIZE=16
ACC_STEPS=8

WD=1e-1

# signSGD without momentum (beta1=0.0): update = -lr * sign(grad)
LR_LIST=(1e-4 5e-4 1e-3 2e-3)

for LR in "${LR_LIST[@]}"; do
    torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
        --distributed-backend nccl \
        --experiment-name "signsgd_nomom_lr_${LR}_wd_${WD}" \
        \
        --dataset fineweb \
        --datasets-dir "${DATASETS_DIR}" \
        --eval-cache-dir "${EVAL_CACHE_DIR}" \
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
        --opt sign_sgd \
        --lr ${LR} \
        --weight-decay ${WD} \
        --beta1 0.0 \
        --grad-clip 1.0 \
        \
        --scheduler cos_warmup_zero \
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
        --wandb-tags baseline bf16 signsgd nomomentum lrsweep
done
