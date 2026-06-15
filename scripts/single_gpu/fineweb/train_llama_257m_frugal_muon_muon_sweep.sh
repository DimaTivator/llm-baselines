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

# ─── Frugal-Muon specific ────────────────────────────────────────────────────
DENSITY=0.25
UPDATE_GAP=50
COORD_CHOICE="columns"
MOMENTUM=0.95      # EMA decay for the stateful (active) Muon subspace
NS_STEPS=5         # Newton-Schulz iterations

# ─── Sweep lists ─────────────────────────────────────────────────────────────
LR_LIST=(1e-3 2e-3)
WD_LIST=(0.1)
DTYPE_LIST=(bfloat16)

# ─── Sweep ────────────────────────────────────────────────────────────────────
for DTYPE in "${DTYPE_LIST[@]}"; do
for LR in "${LR_LIST[@]}"; do
for WD in "${WD_LIST[@]}"; do

    EXP_NAME="llama257M_frugal_muon_muon_${DTYPE}_lr${LR}_wd${WD}_mu${MOMENTUM}_fineweb"
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
        --opt coord_muon \
        --non_proj_opt muon \
        --lr ${LR} \
        --weight-decay ${WD} \
        --momentum ${MOMENTUM} \
        --nesterov \
        --muon_ns_steps ${NS_STEPS} \
        --grad-clip 1.0 \
        \
        --density ${DENSITY} \
        --update_gap ${UPDATE_GAP} \
        --coord_choice ${COORD_CHOICE} \
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
