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

# ─── BAdam-specific ───────────────────────────────────────────────────────────
BLOCK_SIZE=1              # number of transformer layers per trainable block
UPDATE_GAP=50             # steps between block switches
SWITCH_MODE=descending    # random | ascending | descending | fixed
BADAM_VERBOSE=1

# ─── Sweep lists ─────────────────────────────────────────────────────────────
LR_LIST=(1e-4 5e-4 1e-3 2e-3)
WD_LIST=(0.1)
BETA2_LIST=(0.999)
DTYPE_LIST=(bfloat16)

# ─── Sweep ───────────────────────────────────────────────────────────────────
for DTYPE in "${DTYPE_LIST[@]}"; do
for LR in "${LR_LIST[@]}"; do
for WD in "${WD_LIST[@]}"; do
for BETA2 in "${BETA2_LIST[@]}"; do

    EXP_NAME="llama257M_badam_${DTYPE}_lr${LR}_wd${WD}_b2${BETA2}_bs${BLOCK_SIZE}_fineweb"
    echo "==============================================================="
    echo "Starting: ${EXP_NAME}"
    echo "==============================================================="

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
        --dtype ${DTYPE} \
        \
        --opt badam \
        --lr ${LR} \
        --weight-decay ${WD} \
        --beta1 0.9 \
        --beta2 ${BETA2} \
        --grad-clip 1.0 \
        \
        --update_gap ${UPDATE_GAP} \
        --badam_block_size ${BLOCK_SIZE} \
        --badam_switch_mode ${SWITCH_MODE} \
        --badam_verbose ${BADAM_VERBOSE} \
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
