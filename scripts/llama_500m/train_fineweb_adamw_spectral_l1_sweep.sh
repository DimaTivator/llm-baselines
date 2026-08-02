#!/bin/bash
if [ "${MLSUB_CAPTURE_LOG:-0}" = "1" ] && [ "${MLSUB_CAPTURE_ACTIVE:-0}" != "1" ]; then
    LOG_DIR=${MLSUB_LOG_DIR:-"/home/jovyan/logs"}
    mkdir -p "$LOG_DIR"
    MLSUB_PROCESS_RANK=${OMPI_COMM_WORLD_RANK:-0}
    LOG_FILE="$LOG_DIR/llama500m_spectral_l1_rank${MLSUB_PROCESS_RANK}_$(date +%F_%H%M%S).log"
    MLSUB_CAPTURE_ACTIVE=1 bash "$0" "$@" > "$LOG_FILE" 2>&1
    TRAIN_EXIT=$?
    echo "TRAIN_EXIT=$TRAIN_EXIT"
    echo "LOG_FILE=$LOG_FILE"
    echo "=== last 100 lines ==="
    tail -100 "$LOG_FILE"
    exit "$TRAIN_EXIT"
fi

DISTRIBUTED_ARGS=()
TRAIN_LAUNCHER=(python)
if [ "${OMPI_COMM_WORLD_SIZE:-1}" -gt 1 ]; then
    export RANK=${RANK:-"${OMPI_COMM_WORLD_RANK}"}
    export WORLD_SIZE=${WORLD_SIZE:-"${OMPI_COMM_WORLD_SIZE}"}
    export LOCAL_RANK=${LOCAL_RANK:-"${OMPI_COMM_WORLD_LOCAL_RANK:-0}"}
    export MASTER_ADDR=${MASTER_ADDR:-"mpimaster-0"}
    export MASTER_PORT=${MASTER_PORT:-29500}
    DISTRIBUTED_ARGS+=(--distributed_backend nccl)
    echo "DDP rank=${RANK}/${WORLD_SIZE} local_rank=${LOCAL_RANK} master=${MASTER_ADDR}:${MASTER_PORT}"
fi

if [ "${INSTALL_MLSUB_DEPS:-0}" = "1" ]; then
    export PYTHONUSERBASE=${PYTHONUSERBASE:-"/home/jovyan/.local"}
    export PATH="$PYTHONUSERBASE/bin:$PATH"
    python -m pip install --user -q -r requirements-mlsub.txt || exit 1
fi

python -c 'import datasets, huggingface_hub, numpy, tiktoken, torch, tqdm, transformers, wandb, zstandard' \
    || { echo "Missing core training dependencies. Run scripts/install_mlsub_training_deps.sh first."; exit 1; }

DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
TOKENIZED_DATA_DIR=${TOKENIZED_DATA_DIR:-"${HOME}/tokenized_data"}
RESULTS_BASE_FOLDER=${RESULTS_BASE_FOLDER:-"/tmp/llama500m_spectral_l1"}
DEVICE=${DEVICE:-"cuda:0"}
DATASET=${DATASET:-"finewebedu"}

if [ -n "${MLSUB_STUDENT:-}" ] && [ "$DATASET" = "finewebedu" ]; then
    PREPARE_FINEWEBEDU_H200=${PREPARE_FINEWEBEDU_H200:-1}
else
    PREPARE_FINEWEBEDU_H200=${PREPARE_FINEWEBEDU_H200:-0}
fi

if [ "$PREPARE_FINEWEBEDU_H200" = "1" ]; then
    FINEWEBEDU_H200_ROOT=${FINEWEBEDU_H200_ROOT:-"/home/jovyan/finewebedu_h200"}
    python ./src/data/prepare_finewebedu_h200.py --root "$FINEWEBEDU_H200_ROOT" || exit 1
    DATASETS_DIR="$FINEWEBEDU_H200_ROOT/sample/100BT"
    TOKENIZED_DATA_DIR="$FINEWEBEDU_H200_ROOT/tokenized"
fi

if [[ "$DEVICE" == cuda* ]]; then
    GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())') || exit 1
    python -c 'import torch; print(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")' || exit 1
    if [ "$GPU_COUNT" -gt 1 ] && [ "${OMPI_COMM_WORLD_SIZE:-1}" -eq 1 ]; then
        TRAIN_LAUNCHER=(torchrun --standalone --nproc_per_node "$GPU_COUNT")
        DISTRIBUTED_ARGS=(--distributed_backend nccl)
    fi
fi

MODEL_SIZE=${MODEL_SIZE:-"500m"}
OPT="adamw-spectral-l1-reg"
LR=${LR:-1e-3}
N_EMBD=${N_EMBD:-1280}
N_HEAD=${N_HEAD:-20}
N_LAYER=${N_LAYER:-22}
WD=${WD:-1e-1}
ITERATIONS=${ITERATIONS:-76294}
BATCH_SIZE=${BATCH_SIZE:-16}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-1024}
ACC_STEPS=${ACC_STEPS:-8}
WARMUP_STEPS=${WARMUP_STEPS:-2000}
EVAL_BATCHES=${EVAL_BATCHES:-64}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
EVAL_INTERVAL=${EVAL_INTERVAL:-281}
LOG_INTERVAL=${LOG_INTERVAL:-4}
FINEWEBEDU_MAX_FILES=${FINEWEBEDU_MAX_FILES:-5}
EFFECTIVE_RANK_INTERVAL=${EFFECTIVE_RANK_INTERVAL:-500}
DOWNSTREAM_EVAL_INTERVAL=${DOWNSTREAM_EVAL_INTERVAL:-4000}
DOWNSTREAM_EVAL_ENABLED=${DOWNSTREAM_EVAL_ENABLED:-1}
DOWNSTREAM_TASK_GROUP=${DOWNSTREAM_TASK_GROUP:-"basic_v2"}
WANDB_ENABLED=${WANDB_ENABLED:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"ns_weights"}
WANDB_ENTITY=${WANDB_ENTITY:-"andrey"}
HF_CHECKPOINT_REPO=${HF_CHECKPOINT_REPO:-"DimaTivator/effpretrain_ckpts"}
HF_CHECKPOINT_PREFIX=${HF_CHECKPOINT_PREFIX:-"spectral_l1_500m"}
EXPERIMENT_PREFIX=${EXPERIMENT_PREFIX:-""}

if [ "$DOWNSTREAM_EVAL_ENABLED" = "1" ]; then
    python -c 'from olmo_eval import HFTokenizer, ICLMetric, build_task; import cached_path, torchmetrics' \
        || { echo "Missing downstream dependencies. Run scripts/install_mlsub_training_deps.sh first."; exit 1; }
fi

read -r -a SPECTRAL_L1_COEF_VALUES <<< "${SPECTRAL_L1_COEF_START_LIST:-0 0.5 0.7 1 1.2 1.4 1.6 1.8 2 3}"

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
    EXP_NAME="llama${MODEL_SIZE}_${OPT}_wd${WD}_lr${LR}_sl1_${SPECTRAL_L1_COEF_START}_${DATASET}${ERANK_TAG}"
    if [ -n "$EXPERIMENT_PREFIX" ]; then
        EXP_NAME="${EXPERIMENT_PREFIX}_${EXP_NAME}"
    fi
    HF_CHECKPOINT_PATH="${HF_CHECKPOINT_PREFIX}/${EXP_NAME}/ckpts/latest/main.pt"
    echo "=== spectral_l1_reg_coef=${SPECTRAL_L1_COEF_START}  exp=${EXP_NAME} ==="

    "${TRAIN_LAUNCHER[@]}" ./src/main.py \
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
        --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --eval_interval "${EVAL_INTERVAL}" \
        --latest_ckpt_interval 0 \
        --permanent_ckpt_interval 0 \
        --log_interval "${LOG_INTERVAL}" \
        --finewebedu_max_files "${FINEWEBEDU_MAX_FILES}" \
        --tokenized_data_dir "${TOKENIZED_DATA_DIR}" \
        --effective_rank_interval "${EFFECTIVE_RANK_INTERVAL}" \
        --downstream_eval_interval "${DOWNSTREAM_EVAL_INTERVAL}" \
        --downstream_task_group "${DOWNSTREAM_TASK_GROUP}" \
        --hf_checkpoint_repo "${HF_CHECKPOINT_REPO}" \
        --hf_checkpoint_path "${HF_CHECKPOINT_PATH}" \
        "${DISTRIBUTED_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" || exit 1
done
