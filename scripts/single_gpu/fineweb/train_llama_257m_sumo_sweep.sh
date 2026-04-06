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

# ─── SUMO-specific ───────────────────────────────────────────────────────────
DENSITY=0.25              # rank = density * hidden_size per 2-D param
UPDATE_GAP=50             # subspace projection update interval (steps)
SUMO_ALPHA=4.0            # full-space update scale factor
SUMO_GAMMA=1.1            # norm-growth limiter threshold
SUMO_LR_ADAM=1e-4         # AdamW backup lr (for 1-D / non-matrix params)
SUMO_SCALE=1.0            # back-projection scalar multiplier
SUMO_PROJ_TYPE=std        # projector type: std | reverse_std | right | left | full

# ─── Sweep lists ─────────────────────────────────────────────────────────────
LR_LIST=(1e-4 5e-4 1e-3 2e-3)
WD_LIST=(0.1)
MOMENTUM_LIST=(0.95)
DTYPE_LIST=(bfloat16)

# ─── Sweep ───────────────────────────────────────────────────────────────────
for DTYPE in "${DTYPE_LIST[@]}"; do
for LR in "${LR_LIST[@]}"; do
for WD in "${WD_LIST[@]}"; do
for MOMENTUM in "${MOMENTUM_LIST[@]}"; do

    EXP_NAME="llama257M_sumo_${DTYPE}_lr${LR}_wd${WD}_mom${MOMENTUM}_fineweb"
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
        --dtype ${DTYPE} \
        \
        --opt sumo \
        --lr ${LR} \
        --weight-decay ${WD} \
        --beta1 0.9 \
        --beta2 0.999 \
        --grad-clip 1.0 \
        --momentum ${MOMENTUM} \
        --nesterov \
        \
        --density ${DENSITY} \
        --update_gap ${UPDATE_GAP} \
        --sumo_alpha ${SUMO_ALPHA} \
        --sumo_gamma ${SUMO_GAMMA} \
        --sumo_norm_growth_limiter \
        --sumo_lr_adam ${SUMO_LR_ADAM} \
        --sumo_scale ${SUMO_SCALE} \
        --sumo_proj_type ${SUMO_PROJ_TYPE} \
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
