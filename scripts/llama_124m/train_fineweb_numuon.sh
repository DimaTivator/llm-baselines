#!/bin/bash

DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
TOKENIZED_DATA_DIR=${TOKENIZED_DATA_DIR:-"~/tokenized_data"}

MODEL_SIZE="124m"
OPT="numuon"
LR=1e-2
N_EMBD=768
N_HEAD=12
N_LAYER=12
DATASET="finewebedu"
EFFECTIVE_RANK_INTERVAL=500
DOWNSTREAM_EVAL_INTERVAL=2000

# nuMuon-specific: rank fraction anneals from 1.0 --> 0.25 via cosine schedule
# Set RANK_FRACTION_FINAL="" to use a fixed rank fraction instead
RANK_FRACTION=1.0
RANK_FRACTION_FINAL=0.25
RANK_SCHEDULE="cosine"

ERANK_TAG=$([ "$EFFECTIVE_RANK_INTERVAL" -gt 0 ] && echo "_erank" || echo "")
EXP_NAME="llama${MODEL_SIZE}_${OPT}_lr${LR}_rf${RANK_FRACTION}-${RANK_FRACTION_FINAL}_${DATASET}${ERANK_TAG}"

python ./src/main.py \
    --experiment_name "${EXP_NAME}" \
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
    --weight_decay 0.1 \
    --muon_lr_factor "${LR}" \
    --momentum 0.9 \
    --beta1 0.9 --beta2 0.95 \
    --scheduler wsd \
    --warmup_steps 4750 \
    --wsd_fract_decay 0.2 \
    --wsd_final_lr_scale 0.01 \
    --decay_type cosine \
    --dropout 0 \
    --eval_interval 137 \
    --latest_ckpt_interval 1000 \
    --log_interval 4 \
    --finewebedu_max_files 5 \
    --tokenized_data_dir "${TOKENIZED_DATA_DIR}" \
    --effective_rank_interval "${EFFECTIVE_RANK_INTERVAL}" \
    --numuon_rank_fraction "${RANK_FRACTION}" \
    ${RANK_FRACTION_FINAL:+--numuon_rank_fraction_final "${RANK_FRACTION_FINAL}"} \
    --numuon_rank_schedule "${RANK_SCHEDULE}" \
    --numuon_rank_hold_fraction 0.1 \
    --numuon_rank_decay_end_fraction 0.8 \
    --numuon_svd_niter 2 \
    --numuon_svd_oversample 8 \
    --numuon_adamw_lr_factor 0.5 \
    --downstream_eval_enabled \
    --downstream_eval_interval "${DOWNSTREAM_EVAL_INTERVAL}" \
    --downstream_task_group basic_v2 \
    --wandb \
    --wandb_project ns_weights \
    --wandb_entity andrey
