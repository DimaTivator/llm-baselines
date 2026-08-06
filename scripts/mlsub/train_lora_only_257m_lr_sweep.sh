#!/bin/bash
set -uo pipefail

if [ "${MLSUB_CAPTURE_LOG:-0}" = "1" ] && [ "${MLSUB_CAPTURE_ACTIVE:-0}" != "1" ]; then
    LOG_DIR=${MLSUB_LOG_DIR:-"/home/jovyan/logs"}
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/huawei_lora_only_257m_$(date +%F_%H%M%S).log"
    MLSUB_CAPTURE_ACTIVE=1 bash "$0" "$@" > "$LOG_FILE" 2>&1
    TRAIN_EXIT=$?
    echo "TRAIN_EXIT=$TRAIN_EXIT"
    echo "LOG_FILE=$LOG_FILE"
    echo "=== last 120 lines ==="
    tail -120 "$LOG_FILE"
    exit 0
fi

export PYTHONUSERBASE=${PYTHONUSERBASE:-"/home/jovyan/.local"}
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONUNBUFFERED=1

python -c 'import datasets, huggingface_hub, numpy, pyarrow, tiktoken, torch, transformers, wandb' || {
    echo "Missing cached Cloud training dependencies" >&2
    exit 1
}

NGPUS=1
N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256
BATCH_SIZE=16
ACC_STEPS=8
ITERATIONS=${ITERATIONS:-39250}
WARMUP=${WARMUP:-3925}
LORA_RANK=256
LORA_ALPHA=256
LR_LIST_INPUT=${LR_LIST:-"1e-4 5e-4 1e-3 2e-3"}
RESULTS_DIR=${RESULTS_DIR:-"/home/jovyan/exps/huawei-lora-only-257m"}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"/home/jovyan/evals_cache"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}
WANDB_GROUP=${WANDB_GROUP:-"lora-only-rank256-1xC-257m"}

read -r -a LR_VALUES <<< "${LR_LIST_INPUT//,/ }"
if [ "${#LR_VALUES[@]}" -ne 4 ] && [ "${SMOKE_TEST:-0}" != "1" ]; then
    echo "Expected exactly four learning rates, got: ${LR_VALUES[*]}" >&2
    exit 1
fi

if [ "${SMOKE_TEST:-0}" = "1" ]; then
    DATASETS_DIR=/tmp/fineweb-smoke
    mkdir -p "$DATASETS_DIR"
    python - <<'PY'
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

root = Path("/tmp/fineweb-smoke")
for index in range(2):
    path = root / f"smoke_{index}.parquet"
    texts = [f"Small deterministic LoRA smoke-test document {index}-{row}. " * 8 for row in range(100)]
    pq.write_table(pa.table({"text": texts}), path, row_group_size=10)
PY
    N_LAYER=2
    N_EMBD=64
    N_HEAD=4
    SEQ_LEN=64
    MULTIPLE_OF=64
    BATCH_SIZE=1
    ACC_STEPS=1
    ITERATIONS=2
    WARMUP=1
    LORA_RANK=16
    LORA_ALPHA=16
    LR_VALUES=(1e-3)
    DEVICE=cpu
    DTYPE=float32
    LAUNCHER=(python src/main.py)
    EVAL_FLAGS=(--eval-interval 1 --eval-batches 1 --eval-batch-size 1)
    WANDB_FLAGS=()
else
    FINEWEB_ROOT=${FINEWEB_ROOT:-"/workspace-SR006.nfs2/${MLSUB_STUDENT:-dimativator}/huawei-finewebedu-1x-257m"}
    python src/data/prepare_finewebedu_cloud.py --root "$FINEWEB_ROOT" --num-shards 8 || exit 1
    DATASETS_DIR="$FINEWEB_ROOT/sample/100BT"
    DEVICE=cuda:0
    DTYPE=bfloat16
    LAUNCHER=(torchrun --standalone --nproc_per_node=1 src/main.py --distributed-backend nccl)
    EVAL_FLAGS=(
        --eval-interval 500
        --eval-batches 32
        --downstream-eval-enabled
        --downstream-eval-interval 2000
        --downstream-task-group basic_v2
        --lm-eval-enabled
        --lm-eval-interval 2000
        --lm-eval-datasets wikitext103
    )
    WANDB_FLAGS=(--wandb --wandb-project "$WANDB_PROJECT")
fi

mkdir -p "$RESULTS_DIR" "$EVAL_CACHE_DIR"

for LR in "${LR_VALUES[@]}"; do
    EXPERIMENT_NAME="llama257M_lora_only_rank${LORA_RANK}_lr${LR}_1xC"
    echo "=== Starting ${EXPERIMENT_NAME} ==="

    "${LAUNCHER[@]}" \
        --experiment-name "$EXPERIMENT_NAME" \
        --device "$DEVICE" \
        --dataset fineweb \
        --datasets-dir "$DATASETS_DIR" \
        --eval-cache-dir "$EVAL_CACHE_DIR" \
        --sequence-length "$SEQ_LEN" \
        --streaming \
        --workers 8 \
        --model llama \
        --n-layer "$N_LAYER" \
        --n-embd "$N_EMBD" \
        --n-head "$N_HEAD" \
        --multiple-of "$MULTIPLE_OF" \
        --dtype "$DTYPE" \
        --opt lora_pretrain \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --lr "$LR" \
        --weight-decay 0 \
        --beta1 0.9 \
        --beta2 0.999 \
        --grad-clip 1.0 \
        --scheduler cos \
        --warmup-steps "$WARMUP" \
        --iterations "$ITERATIONS" \
        --batch-size "$BATCH_SIZE" \
        --acc-steps "$ACC_STEPS" \
        --log-interval 50 \
        --latest-ckpt-interval 0 \
        --results-base-folder "$RESULTS_DIR" \
        --wandb-group "$WANDB_GROUP" \
        --wandb-tags lora-only bf16 rank-256 257m 1x-chinchilla \
        "${EVAL_FLAGS[@]}" \
        "${WANDB_FLAGS[@]}" || exit 1
done

echo "=== LoRA-only LR sweep completed ==="
