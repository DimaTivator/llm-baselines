#!/bin/bash
set -euo pipefail

GPU=${GPU:-7}
DATASETS_DIR=${DATASETS_DIR:-"/data/datasets/fineweb-edu-100BT/sample/100BT"}
TOKENIZED_DATA_DIR=${TOKENIZED_DATA_DIR:-"$PWD/tokenized"}
RUN_ROOT=${RUN_ROOT:-"$PWD/exps/adamw_l2_svdllm_h200_gpu${GPU}"}
TMP_ROOT="$RUN_ROOT/tmp"
RESULT_ROOT="$RUN_ROOT/results"
read -r -a MODEL_SIZES <<< "${MODEL_SIZES:-124m 257m}"
read -r -a WD_VALUES <<< "${WD_VALUES:-0.1 0.3 0.5 0.7 1 1.5 2}"
read -r -a MARGIN_VALUES <<< "${MARGINS:--8 -4 0 4 8 12 16 20}"

TRAIN_EVAL_BATCHES=${TRAIN_EVAL_BATCHES:-64}
TABLE_EVAL_BATCHES=${TABLE_EVAL_BATCHES:-64}
CALIB_BATCHES=${CALIB_BATCHES:-16}
TABLE_EVAL_BATCH_SIZE=${TABLE_EVAL_BATCH_SIZE:-8}
WARMUP_STEPS=${WARMUP_STEPS:-2000}
WANDB_ENABLED=${WANDB_ENABLED:-1}
TABLE_NO_DOWNSTREAM=${TABLE_NO_DOWNSTREAM:-0}

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export WANDB_BASE_URL=${WANDB_BASE_URL:-"https://wandb-radfan.ru"}
export WANDB_ENTITY=${WANDB_ENTITY:-"andrey"}

mkdir -p "$TMP_ROOT" "$RESULT_ROOT"
test -s "$TOKENIZED_DATA_DIR/train.bin"
test -s "$TOKENIZED_DATA_DIR/val.bin"
python -c 'import cached_path, datasets, numpy, tiktoken, torch, torchmetrics, transformers, wandb, zstandard; from olmo_eval import HFTokenizer, ICLMetric, build_task'
python -c 'import torch; print(f"GPU={torch.cuda.get_device_name(0)} memory={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")'

WANDB_ARGS=()
if [ "$WANDB_ENABLED" = "1" ]; then
    WANDB_ARGS+=(--wandb --wandb_project ns_weights --wandb_entity "$WANDB_ENTITY")
fi

TABLE_EXTRA_ARGS=()
if [ "$TABLE_NO_DOWNSTREAM" = "1" ]; then
    TABLE_EXTRA_ARGS+=(--no_downstream)
fi

CURRENT_EXP_DIR=""
cleanup_current_checkpoint() {
    if [ -z "$CURRENT_EXP_DIR" ]; then
        return
    fi
    case "$CURRENT_EXP_DIR" in
        "$TMP_ROOT"/*) rm -rf -- "$CURRENT_EXP_DIR" ;;
        *) echo "Refusing to remove unexpected path: $CURRENT_EXP_DIR" >&2 ;;
    esac
}
trap cleanup_current_checkpoint EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
    case "$MODEL_SIZE" in
        124m)
            LR=${LR_124M:-5e-3}
            N_EMBD=768
            N_HEAD=12
            N_LAYER=12
            ITERATIONS=${ITERATIONS_124M:-19000}
            EVAL_INTERVAL=${EVAL_INTERVAL_124M:-137}
            BATCH_SIZE=${BATCH_SIZE_124M:-128}
            ACC_STEPS=${ACC_STEPS_124M:-1}
            GLOBAL_BASE_LOSS=${GLOBAL_BASE_LOSS_124M:-3.05995}
            ;;
        257m)
            LR=${LR_257M:-1e-3}
            N_EMBD=1024
            N_HEAD=16
            N_LAYER=16
            ITERATIONS=${ITERATIONS_257M:-39000}
            EVAL_INTERVAL=${EVAL_INTERVAL_257M:-281}
            BATCH_SIZE=${BATCH_SIZE_257M:-64}
            ACC_STEPS=${ACC_STEPS_257M:-2}
            GLOBAL_BASE_LOSS=${GLOBAL_BASE_LOSS_257M:-2.84926}
            ;;
        *)
            echo "Unsupported MODEL_SIZE=$MODEL_SIZE" >&2
            exit 2
            ;;
    esac

    if [ $((BATCH_SIZE * ACC_STEPS)) -ne 128 ]; then
        echo "Expected effective batch 128, got batch_size=$BATCH_SIZE acc_steps=$ACC_STEPS" >&2
        exit 2
    fi

    for WD in "${WD_VALUES[@]}"; do
        OUTPUT_DIR="$RESULT_ROOT/$MODEL_SIZE/wd_$WD"
        if [ -f "$OUTPUT_DIR/COMPLETE" ]; then
            echo "SKIP_COMPLETE model=$MODEL_SIZE wd=$WD output=$OUTPUT_DIR"
            continue
        fi

        EXP_NAME="h200_gpu${GPU}_llama${MODEL_SIZE}_adamw_l2_wd${WD}_lr${LR}_finewebedu"
        CURRENT_EXP_DIR="$TMP_ROOT/$EXP_NAME"
        cleanup_current_checkpoint
        mkdir -p "$OUTPUT_DIR"
        rm -f -- "$OUTPUT_DIR/COMPLETE"

        echo "RUN_START model=$MODEL_SIZE wd=$WD batch_size=$BATCH_SIZE acc_steps=$ACC_STEPS exp=$EXP_NAME"
        python ./src/main.py \
            --experiment_name "$EXP_NAME" \
            --results_base_folder "$TMP_ROOT" \
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
            --batch_size "$BATCH_SIZE" \
            --sequence_length 1024 \
            --acc_steps "$ACC_STEPS" \
            --grad_clip 0.5 \
            --seed 0 \
            --weight_decay "$WD" \
            --scheduler cos \
            --warmup_steps "$WARMUP_STEPS" \
            --dropout 0 \
            --beta1 0.9 \
            --beta2 0.95 \
            --eval_batches "$TRAIN_EVAL_BATCHES" \
            --eval_batch_size 32 \
            --eval_interval "$EVAL_INTERVAL" \
            --latest_ckpt_interval "$ITERATIONS" \
            --permanent_ckpt_interval 0 \
            --log_interval 4 \
            --finewebedu_max_files 5 \
            --tokenized_data_dir "$TOKENIZED_DATA_DIR" \
            --effective_rank_interval 0 \
            "${WANDB_ARGS[@]}"

        CKPT_PATH="$CURRENT_EXP_DIR/ckpts/latest/main.pt"
        test -s "$CKPT_PATH"
        PYTHONPATH=./src python ./src/compression/svdllm_margin_table.py \
            --ckpt_path "$CKPT_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --device cuda:0 \
            --eval_batches "$TABLE_EVAL_BATCHES" \
            --eval_batch_size "$TABLE_EVAL_BATCH_SIZE" \
            --calib_batches "$CALIB_BATCHES" \
            --margins "${MARGIN_VALUES[@]}" \
            --global_base_loss "$GLOBAL_BASE_LOSS" \
            --model_size "$MODEL_SIZE" \
            --weight_decay "$WD" \
            --learning_rate "$LR" \
            "${TABLE_EXTRA_ARGS[@]}"

        test -f "$OUTPUT_DIR/COMPLETE"
        cleanup_current_checkpoint
        CURRENT_EXP_DIR=""
        echo "RUN_COMPLETE model=$MODEL_SIZE wd=$WD output=$OUTPUT_DIR"
    done
done

echo "SWEEP_COMPLETE result_root=$RESULT_ROOT"
