#!/usr/bin/env bash
set -euo pipefail

NGPUS=${1:-1}
DATASETS_DIR="${DATASETS_DIR:-${INPUT_PATH:-./datasets}}"
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"./evals_cache"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain-finewweb-edu-TEST"}
HF_INTER_CKPT_REPO_ID=${HF_INTER_CKPT_REPO_ID:-"fp8-pretrain-checkpoints"}

N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256

ITERATIONS=39250
WARMUP=3925 # 10%
BATCH_SIZE=16
ACC_STEPS=8
LR=1e-4
WEIGHT_DECAY=0.1
BETA2=0.999
DENSITY=0.25
UPDATE_GAP=50

torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
    --distributed-backend nccl \
    --experiment-name "galore_lr1e-4_wd0.1" \
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
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --multiple-of ${MULTIPLE_OF} \
    --dtype bfloat16 \
    \
    --opt galore_adamw \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --beta1 0.9 \
    --beta2 ${BETA2} \
    --grad-clip 1.0 \
    \
    --density ${DENSITY} \
    --update_gap ${UPDATE_GAP} \
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
    --inter-ckpts 20000 \
    --upload-inter-ckpts-to huggingface \
    --hf-inter-ckpt-repo-id "${HF_INTER_CKPT_REPO_ID}" \
    --latest-ckpt-interval 5000 \
    \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-tags galore bf16
