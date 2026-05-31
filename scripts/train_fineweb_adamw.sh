#!/bin/bash

DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
TOKENIZED_DATA_DIR=${TOKENIZED_DATA_DIR:-"~/tokenized_data"}

OPT="adamw-spectral-l1-reg"
LR=1e-3
N_EMBD=768
N_HEAD=12
N_LAYER=12
WD=1e-1
DATASET="finewebedu"
EFFECTIVE_RANK_INTERVAL=500
DOWNSTREAM_EVAL_INTERVAL=2000

ERANK_TAG=$([ "$EFFECTIVE_RANK_INTERVAL" -gt 0 ] && echo "_erank" || echo "")
EXP_NAME="${OPT}_wd${WD}_lr${LR}_embd${N_EMBD}_L${N_LAYER}_${DATASET}${ERANK_TAG}"

python ./src/main.py \
    --experiment_name "${EXP_NAME}" \
    --model llama \
    --datasets_dir "${DATASETS_DIR}" \
    --dataset "${DATASET}" \
    --opt "${OPT}" \
    --lr "${LR}" \
    --iterations 16000 \
    --n_embd "${N_EMBD}" \
    --n_head "${N_HEAD}" \
    --n_layer "${N_LAYER}" \
    --batch_size 64 \
    --sequence_length 512 \
    --acc_steps 4 \
    --grad_clip 0.5 \
    --seed 0 \
    --weight_decay ${WD} \
    --spectral_l1_reg_coef ${WD} \
    --scheduler cos \
    --warmup_steps 2000 \
    --dropout 0 \
    --beta1 0.9 --beta2 0.95 \
    --eval_interval 115 \
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
