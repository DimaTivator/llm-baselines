#!/usr/bin/env bash
set -euo pipefail

NGPUS=${1:-1}
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
# steps = 250M * 20 / (128 * 1024) = 38147
ITERATIONS=38147
WARMUP=3815    # 10% of iterations

# ─── Data ─────────────────────────────────────────────────────────────────────
DATA_PATH=${DATA_PATH:-"/data/datasets/c4-en/en"}

# ─── LORO-specific ───────────────────────────────────────────────────────────
DENSITY=0.25
LORO_TYPE="loro"
LORO_INIT="orth"
LORO_SCOPE="all"
LORO_LR_SCALER=-1.0   # adaptive r/d

# ─── Sweep lists ─────────────────────────────────────────────────────────────
LR_LIST=(1e-4 5e-4 1e-3 2e-3)
WD_LIST=(0.1)
BETA2_LIST=(0.999)
DTYPE_LIST=(bfloat16)

# ─── Sweep ────────────────────────────────────────────────────────────────────
for DTYPE in "${DTYPE_LIST[@]}"; do
for LR in "${LR_LIST[@]}"; do
for WD in "${WD_LIST[@]}"; do
for BETA2 in "${BETA2_LIST[@]}"; do

    EXP_NAME="llama257M_loro_${DTYPE}_lr${LR}_wd${WD}_b2${BETA2}_c4"
    echo "==============================================================="
    echo "Starting: ${EXP_NAME}"
    echo "==============================================================="

    torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
        --distributed-backend nccl \
        \
        --dataset c4 \
        --local_data \
        --local_data_path "${DATA_PATH}" \
        --streaming \
        --sequence-length ${SEQ_LEN} \
        \
        --model llama \
        --n-layer ${N_LAYER} \
        --n-embd  ${N_EMBD} \
        --n-head  ${N_HEAD} \
        --multiple-of ${MULTIPLE_OF} \
        --dtype ${DTYPE} \
        \
        --opt loro \
        --lr ${LR} \
        --weight-decay ${WD} \
        --beta1 0.9 \
        --beta2 ${BETA2} \
        --grad-clip 1.0 \
        \
        --density ${DENSITY} \
        --loro_type ${LORO_TYPE} \
        --loro_init ${LORO_INIT} \
        --loro_scope ${LORO_SCOPE} \
        --loro_lr_scaler ${LORO_LR_SCALER} \
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
