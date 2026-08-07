#!/bin/bash
set -uo pipefail

RANK=${OMPI_COMM_WORLD_RANK:-0}
WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
MODEL_SIZE=${MODEL_SIZE:?MODEL_SIZE must be 124m or 257m}
WD=${WD:?WD must be set}

LOG_ROOT=${LOG_ROOT:-"/home/jovyan/logs/adamw_l2_svdllm"}
RUN_TAG="llama${MODEL_SIZE}_adamw_wd${WD}"
LOG_FILE="${LOG_ROOT}/${RUN_TAG}_rank${RANK}_$(date +%F_%H%M%S).log"

if [ "${MLSUB_CAPTURE_ACTIVE:-0}" != "1" ]; then
    mkdir -p "$LOG_ROOT"
    MLSUB_CAPTURE_ACTIVE=1 bash "$0" "$@" >"$LOG_FILE" 2>&1
    WORKFLOW_EXIT=$?
    echo "WORKFLOW_EXIT=$WORKFLOW_EXIT"
    echo "LOG_FILE=$LOG_FILE"
    echo "=== last 120 lines ==="
    tail -120 "$LOG_FILE"
    # Cloud.ru hides output from failed jobs, so preserve the diagnostic log.
    exit 0
fi

export PYTHONUSERBASE=${PYTHONUSERBASE:-"/home/jovyan/.local"}
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "PREFLIGHT_STAGE=core_imports"
timeout 180 python -c 'import datasets, huggingface_hub, numpy, tiktoken, torch, tqdm, transformers, wandb, zstandard' \
    || { echo "Core dependency preflight failed or timed out" >&2; exit 1; }
echo "PREFLIGHT_STAGE=downstream_imports"
timeout 180 python -c 'from olmo_eval import HFTokenizer, ICLMetric, build_task; import cached_path, torchmetrics' \
    || { echo "Downstream dependency preflight failed or timed out" >&2; exit 1; }
echo "PREFLIGHT_STAGE=data_files"

DATA_ROOT=${FINEWEBEDU_H200_ROOT:-"/home/jovyan/finewebedu_h200"}
DATASETS_DIR="$DATA_ROOT/sample/100BT"
TOKENIZED_DATA_DIR="$DATA_ROOT/tokenized"
test -s "$TOKENIZED_DATA_DIR/train.bin"
test -s "$TOKENIZED_DATA_DIR/val.bin"

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
    echo "PREFLIGHT_OK model=$MODEL_SIZE wd=$WD"
    exit 0
fi

case "$MODEL_SIZE" in
    124m)
        LR=${LR:-5e-3}
        N_EMBD=768
        N_HEAD=12
        N_LAYER=12
        ITERATIONS=${ITERATIONS:-19000}
        EVAL_INTERVAL=${EVAL_INTERVAL:-137}
        GLOBAL_BASE_LOSS=${GLOBAL_BASE_LOSS:-3.05995}
        ;;
    257m)
        LR=${LR:-1e-3}
        N_EMBD=1024
        N_HEAD=16
        N_LAYER=16
        ITERATIONS=${ITERATIONS:-39000}
        EVAL_INTERVAL=${EVAL_INTERVAL:-281}
        GLOBAL_BASE_LOSS=${GLOBAL_BASE_LOSS:-2.84926}
        ;;
    *)
        echo "Unsupported MODEL_SIZE=$MODEL_SIZE" >&2
        exit 2
        ;;
esac

RESULTS_BASE_FOLDER="/tmp/adamw_l2_svdllm_${MODEL_SIZE}_wd${WD}"
EXP_NAME="cloud_llama${MODEL_SIZE}_adamw_l2_wd${WD}_lr${LR}_finewebedu"
EXP_DIR="$RESULTS_BASE_FOLDER/$EXP_NAME"
OUTPUT_DIR=${OUTPUT_DIR:-"/home/jovyan/results/adamw_l2_svdllm/${MODEL_SIZE}/wd_${WD}"}
WARMUP_STEPS=${WARMUP_STEPS:-2000}
TRAIN_EVAL_BATCHES=${TRAIN_EVAL_BATCHES:-64}
TABLE_EVAL_BATCHES=${TABLE_EVAL_BATCHES:-64}
CALIB_BATCHES=${CALIB_BATCHES:-16}
MARGINS_INPUT=${MARGINS:-"-8:-4:0:4:8:12:16:20"}
read -r -a MARGIN_VALUES <<< "${MARGINS_INPUT//:/ }"

DISTRIBUTED_ARGS=()
if [ "$WORLD_SIZE" -gt 1 ]; then
    export RANK
    export WORLD_SIZE
    export LOCAL_RANK
    export MASTER_ADDR=${MASTER_ADDR:-"mpimaster-0"}
    export MASTER_PORT=${MASTER_PORT:-29500}
    DISTRIBUTED_ARGS+=(--distributed_backend nccl)
fi

echo "RUN_TAG=$RUN_TAG rank=$RANK/$WORLD_SIZE local_rank=$LOCAL_RANK"
echo "MODEL_SIZE=$MODEL_SIZE WD=$WD LR=$LR OUTPUT_DIR=$OUTPUT_DIR"
python -c 'import torch; print(f"GPU={torch.cuda.get_device_name(0)} count={torch.cuda.device_count()}")'

python ./src/main.py \
    --experiment_name "$EXP_NAME" \
    --results_base_folder "$RESULTS_BASE_FOLDER" \
    --device cuda:0 \
    --model llama \
    --datasets_dir "$DATASETS_DIR" \
    --dataset finewebedu \
    --opt adamw \
    --lr "$LR" \
    --iterations "$ITERATIONS" \
    --n_embd "$N_EMBD" \
    --n_head "$N_HEAD" \
    --n_layer "$N_LAYER" \
    --batch_size 64 \
    --sequence_length 1024 \
    --acc_steps 2 \
    --grad_clip 0.5 \
    --seed 0 \
    --weight_decay "$WD" \
    --scheduler cos \
    --warmup_steps "$WARMUP_STEPS" \
    --dropout 0 \
    --beta1 0.9 --beta2 0.95 \
    --eval_batches "$TRAIN_EVAL_BATCHES" \
    --eval_batch_size 32 \
    --eval_interval "$EVAL_INTERVAL" \
    --latest_ckpt_interval "$ITERATIONS" \
    --permanent_ckpt_interval 0 \
    --log_interval 4 \
    --finewebedu_max_files 5 \
    --tokenized_data_dir "$TOKENIZED_DATA_DIR" \
    --effective_rank_interval 0 \
    --wandb \
    --wandb_project ns_weights \
    --wandb_entity andrey \
    "${DISTRIBUTED_ARGS[@]}"
TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"
if [ "$TRAIN_EXIT" -ne 0 ]; then
    exit "$TRAIN_EXIT"
fi

if [ "$RANK" -eq 0 ]; then
    cleanup_checkpoint() {
        case "$RESULTS_BASE_FOLDER" in
            /tmp/adamw_l2_svdllm_*) rm -rf -- "$RESULTS_BASE_FOLDER" ;;
            *) echo "Refusing to remove unexpected path: $RESULTS_BASE_FOLDER" >&2 ;;
        esac
    }
    trap cleanup_checkpoint EXIT

    mkdir -p "$OUTPUT_DIR"
    TABLE_EXTRA_ARGS=()
    if [ "${TABLE_NO_DOWNSTREAM:-0}" = "1" ]; then
        TABLE_EXTRA_ARGS+=(--no_downstream)
    fi
    PYTHONPATH=./src python ./src/compression/svdllm_margin_table.py \
        --ckpt_path "$EXP_DIR/ckpts/latest/main.pt" \
        --output_dir "$OUTPUT_DIR" \
        --device cuda:0 \
        --eval_batches "$TABLE_EVAL_BATCHES" \
        --eval_batch_size 8 \
        --calib_batches "$CALIB_BATCHES" \
        --margins "${MARGIN_VALUES[@]}" \
        --global_base_loss "$GLOBAL_BASE_LOSS" \
        --model_size "$MODEL_SIZE" \
        --weight_decay "$WD" \
        --learning_rate "$LR" \
        "${TABLE_EXTRA_ARGS[@]}"
    TABLE_EXIT=$?
    echo "TABLE_EXIT=$TABLE_EXIT"
    exit "$TABLE_EXIT"
fi

exit 0
