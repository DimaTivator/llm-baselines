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
WARMUP=1963
WSD_FRACT_DECAY=0.1
WSD_FINAL_LR_SCALE=0.0
DECAY_TYPE=cosine
BATCH_SIZE=64
ACC_STEPS=2
WEIGHT_DECAY=1e-1

WD=1e-1
MOMENTUM=0.9

# signSGD with momentum (beta1=MOMENTUM):
#   buf = momentum*buf + grad ; update = -lr * sign(buf)
LR_LIST=(1e-3 2e-3)

for LR in "${LR_LIST[@]}"; do
    torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
        --distributed-backend nccl \
        --experiment-name "signsgd_mom${MOMENTUM}_lr_${LR}_wd_${WD}" \
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
        --beta1 ${MOMENTUM} \
        --grad-clip 1.0 \
        \
        --scheduler wsd \
        --warmup-steps ${WARMUP} \
        --iterations ${ITERATIONS} \
        --wsd-fract-decay ${WSD_FRACT_DECAY} \
        --wsd-final-lr-scale ${WSD_FINAL_LR_SCALE} \
        --decay-type ${DECAY_TYPE} \
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
        --wandb-tags baseline bf16 signsgd momentum lrsweep
done
