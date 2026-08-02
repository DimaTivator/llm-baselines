#!/bin/bash
set -eu

export PYTHONUSERBASE=${PYTHONUSERBASE:-"/home/jovyan/.local"}
export PATH="$PYTHONUSERBASE/bin:$PATH"

python -m pip install --user -q -r requirements-mlsub.txt

python -c 'import datasets, huggingface_hub, numpy, tiktoken, torch, tqdm, transformers, wandb, zstandard'
python -c 'from olmo_eval import HFTokenizer, ICLMetric, build_task; import cached_path, torchmetrics'

PYTHONPATH=./src python - <<'PY'
from types import SimpleNamespace

from transformers import AutoTokenizer

from evals.downstream import DownstreamEvaluator

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

cfg = SimpleNamespace(
    downstream_task_group="basic_v2",
    downstream_tasks=None,
    sequence_length=1024,
    eval_batch_size=32,
    device="cpu",
)
evaluator = DownstreamEvaluator(cfg, tokenizer, tokenizer.name_or_path)
evaluator._ensure_initialized()
print("Verified downstream tasks: " + ", ".join(evaluator.task_names))
PY

echo "mlsub training dependencies are ready"
