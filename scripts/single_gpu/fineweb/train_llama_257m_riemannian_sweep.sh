#!/usr/bin/env bash
set -euo pipefail

NGPUS=${1:-1}
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
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

# ─── Riemannian LoRA specific ────────────────────────────────────────────────
# rank = 0 → computed as density * n_embd (e.g. 0.25 * 1024 = 256)
DENSITY=0.25
RIEMANNIAN_RANK=0        # 0 = auto from density
RIEMANNIAN_SCOPE="all"   # all | attn | mlp
RIEMANNIAN_INIT="orth"   # orth (Kaiming-scaled) | zero (adapter style)

# ─── Sweep lists ─────────────────────────────────────────────────────────────
LR_LIST=(1e-4 5e-4 1e-3 2e-3)
WD_LIST=(0.1)
BETA1_LIST=(0.9)
BETA2_LIST=(0.999)
OPT_LIST=(riemannian_adamw)   # riemannian_adamw | riemannian_sgd

# ─── Sweep ───────────────────────────────────────────────────────────────────
for OPT in "${OPT_LIST[@]}"; do
for LR in "${LR_LIST[@]}"; do
for WD in "${WD_LIST[@]}"; do
for BETA1 in "${BETA1_LIST[@]}"; do
for BETA2 in "${BETA2_LIST[@]}"; do

    EXP_NAME="llama257M_${OPT}_scope${RIEMANNIAN_SCOPE}_d${DENSITY}_lr${LR}_wd${WD}_b1${BETA1}_b2${BETA2}_fineweb"
    echo "==============================================================="
    echo "Starting: ${EXP_NAME}"
    echo "==============================================================="

    torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
        --distributed-backend nccl \
        \
        --dataset fineweb \
        --datasets-dir "${DATASETS_DIR}" \
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
        --opt ${OPT} \
        --lr ${LR} \
        --weight-decay ${WD} \
        --beta1 ${BETA1} \
        --beta2 ${BETA2} \
        --grad-clip 1.0 \
        \
        --density ${DENSITY} \
        --riemannian_rank ${RIEMANNIAN_RANK} \
        --riemannian_scope ${RIEMANNIAN_SCOPE} \
        --riemannian_init ${RIEMANNIAN_INIT} \
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
done
