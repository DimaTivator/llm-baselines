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

ITERATIONS=157000
WARMUP=2000
WSD_FRACT_DECAY=0.1
WSD_FINAL_LR_SCALE=0.0
DECAY_TYPE=cosine
INTER_CKPTS=(35325 70650 105975)
BATCH_SIZE=32
ACC_STEPS=4
LR=5e-4
WEIGHT_DECAY=0.1
BETA2=0.99

SLIM_ARGS=()
if [[ -n "${SLIM_ADAM_RULES_JSON:-}" ]]; then
    SLIM_ARGS+=(--slim_adam_rules_json "${SLIM_ADAM_RULES_JSON}")
elif [[ -n "${SLIM_ADAM_LAYER_MAP:-}" ]]; then
    SLIM_ARGS+=(--slim_adam_layer_map "${SLIM_ADAM_LAYER_MAP}")
fi

torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
    --distributed-backend nccl \
    --experiment-name "slim_adam_act_fp8_4xC" \
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
    --opt slim_adam \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --beta1 0.9 \
    --beta2 ${BETA2} \
    --grad-clip 1.0 \
    "${SLIM_ARGS[@]}" \
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
    --fp8 \
    --fp8-fabit E4M3 \
    --fp8-fwbit E4M3 \
    --fp8-babit E5M2 \
    --fp8-bwbit E5M2 \
    --fp8-group-size 16 \
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
    --inter-ckpts "${INTER_CKPTS[@]}" \
    --latest-ckpt-interval 5000 \
    \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group efficient_4xChinchilla_fp8 \
    --wandb-tags fineweb act_fp8 4xChinchilla efficient slim_adam
