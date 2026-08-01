"""Aggregate the real-UCR-dataset label-noise sweep
(scripts/tst_label_noise/train_real_datasets_sweep.sh) into a per-dataset
summary table and a plot, mirroring analyze_noisefrac_sweep.py.

Usage: python src/tst_label_noise/analyze_real_datasets.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "exps" / "tst_real_label_noise"
OUT_PNG = Path(__file__).resolve().parents[2] / "spectral-wd-gen-tst-real-datasets-scan.png"


def load_results() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue
        with open(result_path) as f:
            r = json.load(f)
        dataset = run_dir.name.split("_nf")[0].replace("tst_real_", "")
        rows.append({
            "dataset": dataset,
            "coef": r["spectral_l1_reg_coef"],
            "test_acc": r["final_test_acc"],
            "clean_train_acc": r["clean_label_train_acc"],
            "corrupted_train_acc": r["corrupted_label_train_acc"],
            "mean_erank": r["effective_ranks"]["effective_rank/mean_weighted"],
            "head_erank": r["effective_ranks"]["effective_rank/head"],
        })
    return pd.DataFrame(rows).sort_values(["dataset", "coef"]).reset_index(drop=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset"):
        baseline = g[g["coef"] == 0].iloc[0]
        best = g.loc[g["test_acc"].idxmax()]
        rows.append({
            "dataset": ds,
            "baseline_test_acc": baseline["test_acc"],
            "best_test_acc": best["test_acc"],
            "best_coef": best["coef"],
            "gain_pp": (best["test_acc"] - baseline["test_acc"]) * 100,
        })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ds, g in df.groupby("dataset"):
        axes[0].plot(g["coef"], g["test_acc"], marker="o", label=ds)
        axes[1].plot(g["coef"], g["head_erank"], marker="o", label=ds)
    axes[0].set_xlabel("spectral_l1_reg_coef")
    axes[0].set_ylabel("final test acc (clean)")
    axes[0].set_title("Real UCR datasets: test acc vs coef")
    axes[0].legend()
    axes[1].set_xlabel("spectral_l1_reg_coef")
    axes[1].set_ylabel("head effective rank")
    axes[1].set_title("Classification head effective rank vs coef")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    df = load_results()
    print(df.to_string(index=False))
    summary = summary_table(df)
    print()
    print(summary.to_string(index=False))
    plot(df)
