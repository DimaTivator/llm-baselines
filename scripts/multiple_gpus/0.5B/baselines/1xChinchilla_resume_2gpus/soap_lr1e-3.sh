#!/usr/bin/env bash
set -euo pipefail

NGPUS=${1:-2}
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"./evals_cache"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}

TRUNK_WANDB_GROUP=2xChinchilla_2gpus
TRUNK_EXPERIMENT=soap_lr1e-3_2xC_2gpus
RESUME_STEP=67911
ITERATIONS=75457
RESUME_FROM=${RESUME_FROM:-"${RESULTS_DIR}/${TRUNK_WANDB_GROUP}/${TRUNK_EXPERIMENT}/ckpts/${RESUME_STEP}"}

N_LAYER=18
N_EMBD=1280
N_HEAD=20
SEQ_LEN=1024
MULTIPLE_OF=256

WARMUP=0
WSD_FRACT_DECAY=1.0
WSD_FINAL_LR_SCALE=0.0
DECAY_TYPE=cosine
LATEST_CKPT_INTERVAL=1000
BATCH_SIZE=16
ACC_STEPS=8
LR=1e-3
WEIGHT_DECAY=0.1

torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
    --distributed-backend nccl \
    --experiment-name "soap_lr1e-3_1xC_resume_2gpus" \
    --resume-from "${RESUME_FROM}" \
    --decay-from-checkpoint \
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
    --opt soap \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --beta1 0.9 \
    --beta2 0.99 \
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
    --latest-ckpt-interval ${LATEST_CKPT_INTERVAL} \
    \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group 1xChinchilla_resume_2gpus \
    --wandb-tags baseline bf16 0.5B 2gpus soap resume
