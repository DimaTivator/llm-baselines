"""Label-noise generalization study on real UCR time series classification
datasets (via aeon), reusing the same TimeSeriesTransformer/AdamWSpectralL1Reg
pipeline as train.py -- only the data source changes (real_data.py instead of
the synthetic data.py). wandb logging is intentionally not wired up here:
results are saved locally only (result.json/history.csv), per instruction.

Must be run with `src/` on sys.path, e.g. from the repo root:
    PYTHONPATH=./src python -m tst_label_noise.train_real --dataset_name ECG5000 --noise_frac 0.25 --spectral_l1_reg_coef 1.0
See scripts/tst_label_noise/train_real_datasets_sweep.sh for the full sweep.
"""

import argparse
import dataclasses
import logging
from functools import partial

from .config import TSTLabelNoiseConfig
from .real_data import get_real_loaders, probe_dataset_shape
from .train import train_one_run

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, required=True)
    defaults = dataclasses.asdict(TSTLabelNoiseConfig())
    skip = {"seq_len", "num_classes", "wandb", "run_name"}  # set from the dataset / hardcoded below
    for field in dataclasses.fields(TSTLabelNoiseConfig):
        if field.name in skip:
            continue
        default = defaults[field.name]
        if field.type == bool:
            p.add_argument(f"--{field.name}", action="store_true", default=default)
        else:
            p.add_argument(f"--{field.name}", type=type(default) if default is not None else str, default=default)
    return p.parse_args()


def main():
    args = parse_args()
    seq_len, num_classes = probe_dataset_shape(args.dataset_name)
    logger.info("dataset=%s seq_len=%d num_classes=%d", args.dataset_name, seq_len, num_classes)

    overrides = vars(args).copy()
    del overrides["dataset_name"]
    run_name = f"tst_real_{args.dataset_name}_nf{args.noise_frac}_sl1_{args.spectral_l1_reg_coef}_seed{args.seed}"
    cfg = TSTLabelNoiseConfig(
        **overrides, seq_len=seq_len, num_classes=num_classes, wandb=False, run_name=run_name,
    )
    train_one_run(cfg, loaders_fn=partial(get_real_loaders, dataset_name=args.dataset_name))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
