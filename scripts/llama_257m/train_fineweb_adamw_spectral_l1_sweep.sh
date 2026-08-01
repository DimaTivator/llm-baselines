#!/bin/bash
if [ "${MLSUB_CAPTURE_LOG:-0}" = "1" ] && [ "${MLSUB_CAPTURE_ACTIVE:-0}" != "1" ]; then
    LOG_DIR=${MLSUB_LOG_DIR:-"/home/jovyan/logs"}
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/llama257m_spectral_l1_$(date +%F_%H%M%S).log"
    MLSUB_CAPTURE_ACTIVE=1 bash "$0" "$@" > "$LOG_FILE" 2>&1
    TRAIN_EXIT=$?
    echo "TRAIN_EXIT=$TRAIN_EXIT"
    echo "LOG_FILE=$LOG_FILE"
    echo "=== last 100 lines ==="
    tail -100 "$LOG_FILE"
    exit 0
fi

if [ "${INSTALL_MLSUB_DEPS:-0}" = "1" ]; then
    export PYTHONUSERBASE=${PYTHONUSERBASE:-"/home/jovyan/.local"}
    export PATH="$PYTHONUSERBASE/bin:$PATH"
    python -m pip install --user -q -r requirements-mlsub.txt || exit 1
fi

DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
TOKENIZED_DATA_DIR=${TOKENIZED_DATA_DIR:-"${HOME}/tokenized_data"}
RESULTS_BASE_FOLDER=${RESULTS_BASE_FOLDER:-"./exps/cf_bruteforce_124M"}
DEVICE=${DEVICE:-"cuda:0"}

if [ "${PREPARE_FINEWEBEDU_H200:-0}" = "1" ]; then
    FINEWEBEDU_H200_ROOT=${FINEWEBEDU_H200_ROOT:-"/home/jovyan/finewebedu_h200"}
    python ./src/data/prepare_finewebedu_h200.py --root "$FINEWEBEDU_H200_ROOT" || exit 1
    DATASETS_DIR="$FINEWEBEDU_H200_ROOT/sample/100BT"
    TOKENIZED_DATA_DIR="$FINEWEBEDU_H200_ROOT/tokenized"
fi

if [[ "$DEVICE" == cuda* ]]; then
    python -c 'import torch; print(f"GPU: {torch.cuda.get_device_name(0)}")'
fi

MODEL_SIZE=${MODEL_SIZE:-"257m"}
OPT="adamw-spectral-l1-reg"
read -r -a LR_VALUES <<< "${LR_LIST:-5e-3}"
N_EMBD=${N_EMBD:-1024}
N_HEAD=${N_HEAD:-16}
N_LAYER=${N_LAYER:-16}
WD=${WD:-1e-1}
DATASET=${DATASET:-"finewebedu"}
ITERATIONS=${ITERATIONS:-39000}
BATCH_SIZE=${BATCH_SIZE:-32}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-1024}
ACC_STEPS=${ACC_STEPS:-4}
WARMUP_STEPS=${WARMUP_STEPS:-2000}
EVAL_BATCHES=${EVAL_BATCHES:-64}
EVAL_INTERVAL=${EVAL_INTERVAL:-137}
LATEST_CKPT_INTERVAL=${LATEST_CKPT_INTERVAL:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-4}
FINEWEBEDU_MAX_FILES=${FINEWEBEDU_MAX_FILES:-5}
EFFECTIVE_RANK_INTERVAL=${EFFECTIVE_RANK_INTERVAL:-500}
DOWNSTREAM_EVAL_INTERVAL=${DOWNSTREAM_EVAL_INTERVAL:-2000}
DOWNSTREAM_EVAL_ENABLED=${DOWNSTREAM_EVAL_ENABLED:-1}
WANDB_ENABLED=${WANDB_ENABLED:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"ns_weights"}
WANDB_ENTITY=${WANDB_ENTITY:-"andrey"}
EXPERIMENT_PREFIX=${EXPERIMENT_PREFIX:-"TEST"}

# SPECTRAL_L1_COEF_START_LIST=(0 0.01 0.03 0.05 0.07 0.1 0.3 0.5 0.7 1 2 3 4 5 7)
read -r -a SPECTRAL_L1_COEF_VALUES <<< "${SPECTRAL_L1_COEF_START_LIST:-0}"

SVT_EVERY=0

ERANK_TAG=$([ "$EFFECTIVE_RANK_INTERVAL" -gt 0 ] && echo "_erank" || echo "")

EXTRA_ARGS=()
if [ "$DOWNSTREAM_EVAL_ENABLED" = "1" ]; then
    EXTRA_ARGS+=(--downstream_eval_enabled)
fi
if [ "$WANDB_ENABLED" = "1" ]; then
    EXTRA_ARGS+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY")
fi

for SPECTRAL_L1_COEF_START in "${SPECTRAL_L1_COEF_VALUES[@]}"; do
for LR in "${LR_VALUES[@]}"; do
    EXP_NAME="${EXPERIMENT_PREFIX}_llama${MODEL_SIZE}_${OPT}_wd${WD}_lr${LR}_sl1_${SPECTRAL_L1_COEF_START}_${DATASET}${ERANK_TAG}"
    echo "=== spectral_l1_reg_coef=${SPECTRAL_L1_COEF_START}  exp=${EXP_NAME} ==="

    python ./src/main.py \
        --experiment_name "${EXP_NAME}" \
        --results_base_folder "${RESULTS_BASE_FOLDER}" \
        --device "${DEVICE}" \
        --model llama \
        --datasets_dir "${DATASETS_DIR}" \
        --dataset "${DATASET}" \
        --opt "${OPT}" \
        --lr "${LR}" \
        --iterations "${ITERATIONS}" \
        --n_embd "${N_EMBD}" \
        --n_head "${N_HEAD}" \
        --n_layer "${N_LAYER}" \
        --batch_size "${BATCH_SIZE}" \
        --sequence_length "${SEQUENCE_LENGTH}" \
        --acc_steps "${ACC_STEPS}" \
        --grad_clip 0.5 \
        --seed 0 \
        --weight_decay "${WD}" \
        --spectral_l1_reg_coef "${SPECTRAL_L1_COEF_START}" \
        --spectral_l1_svt_interval "${SVT_EVERY}" \
        --scheduler cos \
        --warmup_steps "${WARMUP_STEPS}" \
        --dropout 0 \
        --beta1 0.9 --beta2 0.95 \
        --eval_batches "${EVAL_BATCHES}" \
        --eval_interval "${EVAL_INTERVAL}" \
        --latest_ckpt_interval "${LATEST_CKPT_INTERVAL}" \
        --log_interval "${LOG_INTERVAL}" \
        --finewebedu_max_files "${FINEWEBEDU_MAX_FILES}" \
        --tokenized_data_dir "${TOKENIZED_DATA_DIR}" \
        --effective_rank_interval "${EFFECTIVE_RANK_INTERVAL}" \
        --downstream_eval_interval "${DOWNSTREAM_EVAL_INTERVAL}" \
        --downstream_task_group basic_v2 \
        "${EXTRA_ARGS[@]}"
done
done
