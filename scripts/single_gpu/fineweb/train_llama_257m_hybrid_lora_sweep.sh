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
BATCH_SIZE=16
ACC_STEPS=8    # 16 * 8 * 1 = 128
ITERATIONS=39250
WARMUP=3925    # 10% of iterations

# ─── HybridLoRA-specific ─────────────────────────────────────────────────────
# rank = 0 -> auto-computed as density * n_embd (e.g. 0.25 * 1024 = 256)
# DENSITY=0.25
HYBRID_LORA_RANK=32         # 0 = auto from density
HYBRID_LORA_ALPHA=1.0
HYBRID_LORA_SCOPE="all"     # all | attn | mlp
HYBRID_LORA_LR_SCALE=1.0    # LoRA adapter LR = base LR * this factor

# Leave empty to target all Linear layers in HYBRID_LORA_SCOPE.
HYBRID_LORA_TARGET_MODULES=""

# ─── Sweep lists ─────────────────────────────────────────────────────────────
LR_LIST=(1e-4 5e-4 1e-3 2e-3)
WD_LIST=(0.1)
BETA1_LIST=(0.9)
BETA2_LIST=(0.999)
BASE_OPT_LIST=(sgd)
LORA_OPT_LIST=(riemannian_sgd)

# ─── Sweep ───────────────────────────────────────────────────────────────────
for BASE_OPT in "${BASE_OPT_LIST[@]}"; do
for LORA_OPT in "${LORA_OPT_LIST[@]}"; do
for LR in "${LR_LIST[@]}"; do
for WD in "${WD_LIST[@]}"; do
for BETA1 in "${BETA1_LIST[@]}"; do
for BETA2 in "${BETA2_LIST[@]}"; do

    EXP_NAME="llama257M_hybrid_lora_base${BASE_OPT}_lora${LORA_OPT}_scope${HYBRID_LORA_SCOPE}_d${DENSITY}_lr${LR}_wd${WD}_b1${BETA1}_b2${BETA2}_fineweb"
    echo "==============================================================="
    echo "Starting: ${EXP_NAME}"
    echo "==============================================================="

    # Build optional target-modules flag.
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
        --density ${DENSITY} \
        --hybrid_lora_rank ${HYBRID_LORA_RANK} \
        --hybrid_lora_alpha ${HYBRID_LORA_ALPHA} \
        --hybrid_lora_scope ${HYBRID_LORA_SCOPE} \
        --hybrid_lora_base_opt ${BASE_OPT} \
        --hybrid_lora_lora_opt ${LORA_OPT} \
        --hybrid_lora_lora_lr_scale ${HYBRID_LORA_LR_SCALE} \
        ${TARGET_MODULES_FLAG} \
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
        --experiment-name "${EXP_NAME}" \
        --results-base-folder "${RESULTS_DIR}" \
        --wandb \
        --wandb-project "${WANDB_PROJECT}"

done
done
done
done
done
done
