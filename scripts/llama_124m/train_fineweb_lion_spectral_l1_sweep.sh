#!/bin/bash
set -euo pipefail

DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
TOKENIZED_DATA_DIR=${TOKENIZED_DATA_DIR:-"./tokenized"}
GPU=${GPU:-1}

MODEL_SIZE="124m"
OPT="lion-spectral-l1-reg"
LR_LIST=(1e-3)
N_EMBD=768
N_HEAD=12
N_LAYER=12
WD=1e-1
DATASET="finewebedu"
EFFECTIVE_RANK_INTERVAL=500
DOWNSTREAM_EVAL_INTERVAL=2000
read -r -a SPECTRAL_L1_COEF_START_LIST <<< "${SPECTRAL_L1_COEFS:-0 0.1 0.5 1}"
SVT_EVERY=0

ERANK_TAG=$([ "$EFFECTIVE_RANK_INTERVAL" -gt 0 ] && echo "_erank" || echo "")

for SPECTRAL_L1_COEF_START in "${SPECTRAL_L1_COEF_START_LIST[@]}"; do
for LR in "${LR_LIST[@]}"; do
    EXP_NAME="llama${MODEL_SIZE}_${OPT}_wd${WD}_lr${LR}_sl1_${SPECTRAL_L1_COEF_START}_${DATASET}${ERANK_TAG}"
    echo "=== spectral_l1_reg_coef=${SPECTRAL_L1_COEF_START}  exp=${EXP_NAME} ==="

    CUDA_VISIBLE_DEVICES="${GPU}" python ./src/main.py \
        --experiment_name "${EXP_NAME}" \
        --results_base_folder "./exps/cf_bruteforce_lion_124M" \
        --model llama \
        --datasets_dir "${DATASETS_DIR}" \
        --dataset "${DATASET}" \
        --opt "${OPT}" \
        --lr "${LR}" \
        --iterations 19000 \
        --n_embd "${N_EMBD}" \
        --n_head "${N_HEAD}" \
        --n_layer "${N_LAYER}" \
        --batch_size 64 \
        --sequence_length 1024 \
        --acc_steps 2 \
        --grad_clip 0.5 \
        --seed 0 \
        --weight_decay "${WD}" \
        --spectral_l1_reg_coef "${SPECTRAL_L1_COEF_START}" \
        --spectral_l1_svt_interval "${SVT_EVERY}" \
        --scheduler cos \
        --warmup_steps 2000 \
        --dropout 0 \
        --beta1 0.9 --beta2 0.95 \
        --eval_interval 137 \
        --latest_ckpt_interval 1000 \
        --log_interval 4 \
        --finewebedu_max_files 5 \
        --tokenized_data_dir "${TOKENIZED_DATA_DIR}" \
        --effective_rank_interval "${EFFECTIVE_RANK_INTERVAL}" \
        --downstream_eval_enabled \
        --downstream_eval_interval "${DOWNSTREAM_EVAL_INTERVAL}" \
        --downstream_task_group basic_v2 \
        --wandb \
        --wandb_project ns_weights \
        --wandb_entity andrey
done
done
