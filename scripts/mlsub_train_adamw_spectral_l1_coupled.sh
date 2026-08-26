#!/bin/bash
set -uo pipefail

CF=${CF:-1}
WD=${WD:-1e-1}
LR=${LR:-5e-3}
ITERATIONS=${ITERATIONS:-19000}
LOG_ROOT=${LOG_ROOT:-"/home/jovyan/logs/adamw_spectral_l1_coupled"}
RUN_TAG="llama124m_adamw_spectral_l1_coupled_cf${CF}_lr${LR}"
LOG_FILE="${LOG_ROOT}/${RUN_TAG}_$(date +%F_%H%M%S).log"

if [ "${MLSUB_CAPTURE_ACTIVE:-0}" != "1" ]; then
    mkdir -p "$LOG_ROOT"
    MLSUB_CAPTURE_ACTIVE=1 bash "$0" "$@" >"$LOG_FILE" 2>&1
    WORKFLOW_EXIT=$?
    echo "WORKFLOW_EXIT=$WORKFLOW_EXIT"
    echo "LOG_FILE=$LOG_FILE"
    echo "=== last 120 lines ==="
    tail -120 "$LOG_FILE"
    # Cloud.ru hides logs for failed jobs, so return zero after persisting diagnostics.
    exit 0
fi

export PYTHONUSERBASE=${PYTHONUSERBASE:-"/home/jovyan/.local"}
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "INSTALL_STAGE=training_and_downstream_dependencies"
bash ./scripts/install_mlsub_training_deps.sh || exit $?

DATA_ROOT=${FINEWEBEDU_ROOT:-"/home/jovyan/finewebedu_h200"}
DATASETS_DIR="$DATA_ROOT/sample/100BT"
TOKENIZED_DATA_DIR="$DATA_ROOT/tokenized"
test -s "$TOKENIZED_DATA_DIR/train.bin"
test -s "$TOKENIZED_DATA_DIR/val.bin"

EXP_NAME="cloud_llama124m_adamw_spectral_l1_coupled_wd${WD}_lr${LR}_cf${CF}_finewebedu_erank"
RESULTS_BASE_FOLDER="/tmp/adamw_spectral_l1_coupled_cf${CF}"

echo "RUN_START exp=$EXP_NAME cf=$CF lr=$LR wd=$WD iterations=$ITERATIONS"
python -c 'import torch; print(f"GPU={torch.cuda.get_device_name(0)} count={torch.cuda.device_count()}")'

python ./src/main.py \
    --experiment_name "$EXP_NAME" \
    --results_base_folder "$RESULTS_BASE_FOLDER" \
    --device cuda:0 \
    --model llama \
    --datasets_dir "$DATASETS_DIR" \
    --dataset finewebedu \
    --opt adamw-spectral-l1-reg \
    --lr "$LR" \
    --iterations "$ITERATIONS" \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 64 \
    --sequence_length 1024 \
    --acc_steps 2 \
    --grad_clip 0.5 \
    --seed 0 \
    --weight_decay "$WD" \
    --spectral_l1_reg_coef "$CF" \
    --spectral_l1_reg_coupled \
    --spectral_l1_svt_interval 0 \
    --scheduler cos \
    --warmup_steps 2000 \
    --dropout 0 \
    --beta1 0.9 --beta2 0.95 \
    --eval_batches 64 \
    --eval_batch_size 32 \
    --eval_interval 137 \
    --latest_ckpt_interval "$ITERATIONS" \
    --permanent_ckpt_interval 0 \
    --log_interval 4 \
    --finewebedu_max_files 5 \
    --tokenized_data_dir "$TOKENIZED_DATA_DIR" \
    --effective_rank_interval 500 \
    --downstream_eval_enabled \
    --downstream_eval_interval 2000 \
    --downstream_task_group basic_v2 \
    --wandb \
    --wandb_project ns_weights \
    --wandb_entity andrey
TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"
exit "$TRAIN_EXIT"
