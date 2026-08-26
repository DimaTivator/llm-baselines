#!/bin/bash
set -uo pipefail

ITERATIONS=${ITERATIONS:-19000}
MATRIX_LR=${MATRIX_LR:-1e-2}
SMOKE=${SMOKE:-0}
ENABLE_DOWNSTREAM=${ENABLE_DOWNSTREAM:-1}
LOG_ROOT=${LOG_ROOT:-"/home/jovyan/logs/numuon"}
RUN_TAG="llama124m_numuon_lr${MATRIX_LR}_steps${ITERATIONS}"
LOG_FILE="${LOG_ROOT}/${RUN_TAG}_$(date +%F_%H%M%S).log"

if [ "${MLSUB_CAPTURE_ACTIVE:-0}" != "1" ]; then
    mkdir -p "$LOG_ROOT"
    MLSUB_CAPTURE_ACTIVE=1 bash "$0" "$@" >"$LOG_FILE" 2>&1
    WORKFLOW_EXIT=$?
    echo "WORKFLOW_EXIT=$WORKFLOW_EXIT"
    echo "LOG_FILE=$LOG_FILE"
    echo "=== last 160 lines ==="
    tail -160 "$LOG_FILE"
    # Failed Cloud jobs hide stdout. Preserve the real exit status in the log,
    # while letting mlsub expose the captured diagnostics.
    exit 0
fi

export PYTHONUSERBASE=${PYTHONUSERBASE:-"/home/jovyan/.local"}
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "INSTALL_STAGE=training_and_downstream_dependencies"
bash ./scripts/install_mlsub_training_deps.sh || exit $?

echo "PREFLIGHT_STAGE=numuon_cuda_math"
PYTHONPATH=./src python - <<'PY'
import torch

from optim.numuon import _block_krylov_svd, _current_rank_fraction

device = torch.device("cuda:0")
torch.manual_seed(0)
matrix = torch.randn(32, 24, device=device)
u, _, v = _block_krylov_svd(matrix, 6, L=2, oversample=8)
singular_values = torch.linalg.svdvals(u @ v.T)
assert torch.allclose(singular_values[:6], torch.ones(6, device=device), atol=3e-3)
assert singular_values[6] < 3e-3
assert _current_rank_fraction(1.0, 0.25, "cosine", 100, 1000) == 1.0
assert _current_rank_fraction(1.0, 0.25, "cosine", 800, 1000) == 0.25
print("NUMUON_CUDA_PREFLIGHT=OK")
PY

DATA_ROOT=${FINEWEBEDU_ROOT:-"/home/jovyan/finewebedu_h200"}
DATASETS_DIR="$DATA_ROOT/sample/100BT"
TOKENIZED_DATA_DIR="$DATA_ROOT/tokenized"
test -s "$TOKENIZED_DATA_DIR/train.bin"
test -s "$TOKENIZED_DATA_DIR/val.bin"

WARMUP_STEPS=$((ITERATIONS / 4))
if [ "$WARMUP_STEPS" -lt 1 ]; then
    WARMUP_STEPS=1
fi

EXP_NAME="cloud_llama124m_numuon_paper_lr${MATRIX_LR}_rf1.0-0.25_finewebedu_erank"
if [ "$SMOKE" = "1" ]; then
    EXP_NAME="${EXP_NAME}_smoke${ITERATIONS}"
fi

EXTRA_ARGS=()
if [ "$ENABLE_DOWNSTREAM" = "1" ]; then
    EXTRA_ARGS+=(
        --downstream_eval_enabled
        --downstream_eval_interval 2000
        --downstream_task_group basic_v2
    )
fi

echo "RUN_START exp=$EXP_NAME iterations=$ITERATIONS matrix_lr=$MATRIX_LR"
python -c 'import torch; print(f"GPU={torch.cuda.get_device_name(0)} count={torch.cuda.device_count()}")'

python ./src/main.py \
    --experiment_name "$EXP_NAME" \
    --results_base_folder /tmp/numuon_124m \
    --device cuda:0 \
    --model llama \
    --datasets_dir "$DATASETS_DIR" \
    --dataset finewebedu \
    --opt numuon \
    --lr "$MATRIX_LR" \
    --muon_lr_factor "$MATRIX_LR" \
    --iterations "$ITERATIONS" \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 64 \
    --sequence_length 1024 \
    --acc_steps 2 \
    --grad_clip 1.0 \
    --seed 0 \
    --weight_decay 0.1 \
    --momentum 0.9 \
    --beta1 0.9 --beta2 0.95 \
    --scheduler wsd \
    --warmup_steps "$WARMUP_STEPS" \
    --wsd_fract_decay 0.2 \
    --wsd_final_lr_scale 0.01 \
    --decay_type cosine \
    --dropout 0 \
    --eval_batches 64 \
    --eval_batch_size 32 \
    --eval_interval 137 \
    --latest_ckpt_interval 0 \
    --permanent_ckpt_interval 0 \
    --log_interval 1 \
    --finewebedu_max_files 5 \
    --tokenized_data_dir "$TOKENIZED_DATA_DIR" \
    --effective_rank_interval 500 \
    --numuon_rank_fraction 1.0 \
    --numuon_rank_fraction_final 0.25 \
    --numuon_rank_schedule cosine \
    --numuon_rank_hold_fraction 0.1 \
    --numuon_rank_decay_end_fraction 0.8 \
    --numuon_svd_niter 2 \
    --numuon_svd_oversample 8 \
    --numuon_adamw_lr_factor 0.5 \
    "${EXTRA_ARGS[@]}" \
    --wandb \
    --wandb_project ns_weights \
    --wandb_entity andrey
TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"
exit "$TRAIN_EXIT"
