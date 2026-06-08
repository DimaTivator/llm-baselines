#!/usr/bin/env bash
set -euo pipefail

NGPUS=${1:-1}
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"./evals_cache"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}

# ─── Model (Llama-257M: 12L/1024D/8H) ───────────────────────────────────────
N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256

# ─── Training constants ──────────────────────────────────────────────────────
# total_batch_size = batch_size * acc_steps * ngpus = 128
BATCH_SIZE=32
ACC_STEPS=4    # 16 * 8 * 1 = 128
ITERATIONS=39250
WARMUP=3925    # 10% of iterations

# ─── HybridLoRA-specific ─────────────────────────────────────────────────────
HYBRID_LORA_RANK=32
HYBRID_LORA_ALPHA=1.0
HYBRID_LORA_SCOPE="all"
HYBRID_LORA_LR_SCALE=1.0

HYBRID_LORA_TARGET_MODULES=""

# ─── WSD scheduler ───────────────────────────────────────────────────────────
WSD_FRACT_DECAY=0.1
WSD_FINAL_LR_SCALE=0.0
DECAY_TYPE=cosine

BASE_OPT="sign_sgd"
LORA_OPT="adamw"

# ─── Sweep lists ─────────────────────────────────────────────────────────────
LR_LIST=(1e-3)
WD_LIST=(0.1)
BETA1_LIST=(0.9)
BETA2_LIST=(0.999)

# ─── Sweep ───────────────────────────────────────────────────────────────────
for LR in "${LR_LIST[@]}"; do
for WD in "${WD_LIST[@]}"; do
for BETA1 in "${BETA1_LIST[@]}"; do
for BETA2 in "${BETA2_LIST[@]}"; do

    EXP_NAME="llama257M_hybrid_lora_base_${BASE_OPT}_lora_${LORA_OPT}_scope${HYBRID_LORA_SCOPE}_rank${HYBRID_LORA_RANK}_lr${LR}_wd${WD}_b1${BETA1}_b2${BETA2}_fineweb"
    echo "==============================================================="
    echo "Starting: ${EXP_NAME}"
    echo "==============================================================="

    TARGET_MODULES_FLAG=""
    if [[ -n "${HYBRID_LORA_TARGET_MODULES}" ]]; then
        TARGET_MODULES_FLAG="--hybrid_lora_target_modules ${HYBRID_LORA_TARGET_MODULES}"
    fi

    torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
        --distributed-backend nccl \
        \
        --dataset fineweb \
        --datasets-dir "${DATASETS_DIR}" \
        --eval-cache-dir "${EVAL_CACHE_DIR}" \
        --streaming \
        --workers 8 \
        --sequence-length ${SEQ_LEN} \
        \
        --model llama \
        --n-layer ${N_LAYER} \
        --n-embd  ${N_EMBD} \
        --n-head  ${N_HEAD} \
        --multiple-of ${MULTIPLE_OF} \
        --dtype bfloat16 \
        \
        --opt hybrid_lora \
        --lr ${LR} \
        --weight-decay ${WD} \
        --beta1 ${BETA1} \
        --beta2 ${BETA2} \
        --grad-clip 1.0 \
        \
        --hybrid_lora_rank ${HYBRID_LORA_RANK} \
        --hybrid_lora_alpha ${HYBRID_LORA_ALPHA} \
        --hybrid_lora_scope ${HYBRID_LORA_SCOPE} \
        --hybrid_lora_base_opt ${BASE_OPT} \
        --hybrid_lora_lora_opt ${LORA_OPT} \
        --hybrid_lora_lora_lr_scale ${HYBRID_LORA_LR_SCALE} \
        ${TARGET_MODULES_FLAG} \
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
        --experiment-name "${EXP_NAME}" \
        --results-base-folder "${RESULTS_DIR}" \
        --wandb \
        --wandb-project "${WANDB_PROJECT}"

done
done
done
done
