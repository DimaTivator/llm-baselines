#!/bin/bash
set -euo pipefail

GPU=${GPU:?Set the physical GPU index}
DECAY_TYPE=${DECAY_TYPE:?Set DECAY_TYPE to l2 or spectral}
read -r -a COEFFICIENT_VALUES <<< "${COEFFICIENTS:?Set space-separated COEFFICIENTS}"

DATASETS_DIR=${DATASETS_DIR:-"/data/users/dimativator/llm-baselines-soap/datasets"}
TOKENIZED_DATA_DIR=${TOKENIZED_DATA_DIR:-"/data/users/dimativator/llm-baselines-soap/tokenized"}
RUN_ROOT=${RUN_ROOT:-"$PWD/exps/galore_124m_h200"}
LR=${LR:-1e-3}
DENSITY=${DENSITY:-0.25}
UPDATE_PROJ_GAP=${UPDATE_PROJ_GAP:-200}
GALORE_SCALE=${GALORE_SCALE:-0.25}
ITERATIONS=${ITERATIONS:-19000}
PYTHON_BIN=${PYTHON_BIN:-python}

case "$DECAY_TYPE" in
    l2|spectral) ;;
    *) echo "DECAY_TYPE must be l2 or spectral" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export WANDB_BASE_URL=${WANDB_BASE_URL:-"https://wandb-radfan.ru"}

test -s "$TOKENIZED_DATA_DIR/train.bin"
test -s "$TOKENIZED_DATA_DIR/val.bin"
mkdir -p "$RUN_ROOT"

"$PYTHON_BIN" -c 'import torch; print(f"CUDA_PREFLIGHT=OK gpu={torch.cuda.get_device_name(0)} free={torch.cuda.mem_get_info()[0] / 2**30:.1f}GiB")'

for COEFFICIENT in "${COEFFICIENT_VALUES[@]}"; do
    EXP_NAME="h200_gpu${GPU}_llama124m_galore_${DECAY_TYPE}_cf${COEFFICIENT}_lr${LR}_density${DENSITY}_finewebedu"
    EXP_DIR="$RUN_ROOT/$EXP_NAME"
    if [ -f "$EXP_DIR/COMPLETE" ]; then
        echo "SKIP_COMPLETE decay_type=$DECAY_TYPE cf=$COEFFICIENT"
        continue
    fi

    WEIGHT_DECAY=0
    SPECTRAL_COEFFICIENT=0
    if [ "$DECAY_TYPE" = "l2" ]; then
        WEIGHT_DECAY=$COEFFICIENT
    else
        SPECTRAL_COEFFICIENT=$COEFFICIENT
    fi

    echo "RUN_START decay_type=$DECAY_TYPE cf=$COEFFICIENT exp=$EXP_NAME"
    "$PYTHON_BIN" ./src/main.py \
        --experiment_name "$EXP_NAME" \
        --results_base_folder "$RUN_ROOT" \
        --device cuda:0 \
        --model llama \
        --datasets_dir "$DATASETS_DIR" \
        --dataset finewebedu \
        --opt galore \
        --galore_density "$DENSITY" \
        --galore_update_proj_gap "$UPDATE_PROJ_GAP" \
        --galore_scale "$GALORE_SCALE" \
        --galore_weight_decay_type "$DECAY_TYPE" \
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
        --weight_decay "$WEIGHT_DECAY" \
        --spectral_l1_reg_coef "$SPECTRAL_COEFFICIENT" \
        --scheduler cos \
        --warmup_steps 2000 \
        --dropout 0 \
        --beta1 0.9 \
        --beta2 0.95 \
        --eval_interval 137 \
        --eval_batches 64 \
        --eval_batch_size 64 \
        --latest_ckpt_interval 1000 \
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

    touch "$EXP_DIR/COMPLETE"
    echo "RUN_COMPLETE decay_type=$DECAY_TYPE cf=$COEFFICIENT exp=$EXP_NAME"
done

echo "SWEEP_COMPLETE decay_type=$DECAY_TYPE gpu=$GPU"
