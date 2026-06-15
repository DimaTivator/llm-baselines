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
WEIGHT_DECAY=0.1

# ─── Frugal-Muon specific ────────────────────────────────────────────────────
DENSITY=0.25
UPDATE_GAP=50
COORD_CHOICE="columns"
MOMENTUM=0.95
NS_STEPS=5

LR_LIST=(1e-3)

for LR in "${LR_LIST[@]}"; do

    EXP_NAME="llama257M_muon_muon_lr${LR}_fineweb"
    echo "==============================================================="
    echo "Starting: ${EXP_NAME}"
    echo "==============================================================="

    torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
        --distributed-backend nccl \
        --experiment-name "${EXP_NAME}" \
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
        --opt coord_muon \
        --non_proj_opt muon \
        --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --momentum ${MOMENTUM} \
        --nesterov \
        --muon_ns_steps ${NS_STEPS} \
        --grad-clip 1.0 \
        \
        --density ${DENSITY} \
        --update_gap ${UPDATE_GAP} \
        --coord_choice ${COORD_CHOICE} \
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
        --wandb-project "${WANDB_PROJECT}"

done
