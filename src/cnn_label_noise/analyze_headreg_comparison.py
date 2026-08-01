"""Compares the head-excluded (§10, exps/cnn_label_noise/) and head-included
(exps/cnn_label_noise_headreg/, --include_head_in_reg) variants of the
CNN-on-CIFAR10 label-noise sweep, to test §10's open question: is the
classifier head the bottleneck for spectral WD's noise-memorization-blocking
effect?

Usage: python src/cnn_label_noise/analyze_headreg_comparison.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_DIR = REPO_ROOT / "exps" / "cnn_label_noise"
INCLUDED_DIR = REPO_ROOT / "exps" / "cnn_label_noise_headreg"
OUT_PNG = REPO_ROOT / "spectral-wd-gen-cnn-headreg-comparison.png"


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(results_dir.iterdir()):
        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue
        with open(result_path) as f:
            r = json.load(f)
        rows.append({
            "noise_frac": r["noise_frac"],
            "coef": r["spectral_l1_reg_coef"],
            "test_acc": r["final_test_acc"],
            "clean_train_acc": r["clean_label_train_acc"],
            "corrupted_train_acc": r["corrupted_label_train_acc"],
            "mean_erank": r["effective_ranks"]["effective_rank/mean_weighted"],
            "head_erank": r["effective_ranks"]["effective_rank/head"],
        })
    return pd.DataFrame(rows).sort_values(["noise_frac", "coef"]).reset_index(drop=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for nf, g in df.groupby("noise_frac"):
        baseline = g[g["coef"] == 0].iloc[0]
        best = g.loc[g["test_acc"].idxmax()]
        rows.append({
            "noise_frac": nf,
            "baseline_test_acc": baseline["test_acc"],
            "best_test_acc": best["test_acc"],
            "best_coef": best["coef"],
            "gain_pp": (best["test_acc"] - baseline["test_acc"]) * 100,
        })
    return pd.DataFrame(rows)


def plot(excluded: pd.DataFrame, included: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for nf in sorted(excluded["noise_frac"].unique()):
        ge = excluded[excluded["noise_frac"] == nf]
        gi = included[included["noise_frac"] == nf]
        axes[0].plot(ge["coef"], ge["test_acc"], marker="o", linestyle="-", label=f"nf={nf} head excl.")
        axes[0].plot(gi["coef"], gi["test_acc"], marker="s", linestyle="--", label=f"nf={nf} head incl.")
        axes[1].plot(ge["coef"], ge["corrupted_train_acc"], marker="o", linestyle="-", label=f"nf={nf} head excl.")
        axes[1].plot(gi["coef"], gi["corrupted_train_acc"], marker="s", linestyle="--", label=f"nf={nf} head incl.")
        axes[2].plot(gi["coef"], gi["head_erank"], marker="s", linestyle="--", label=f"nf={nf} head incl.")
    axes[0].set_xlabel("spectral_l1_reg_coef")
    axes[0].set_ylabel("final test acc")
    axes[0].set_title("Test acc: head-excluded vs head-included")
    axes[0].legend(fontsize=6)
    axes[1].set_xlabel("spectral_l1_reg_coef")
    axes[1].set_ylabel("corrupted-label train acc")
    axes[1].set_title("Noise memorization: head-excluded vs head-included")
    axes[1].legend(fontsize=6)
    axes[2].set_xlabel("spectral_l1_reg_coef")
    axes[2].set_ylabel("head effective rank")
    axes[2].set_title("Head effective rank (head-included only, max=10)")
    axes[2].set_ylim(0, 10.5)
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    excluded = load_results(EXCLUDED_DIR)
    included = load_results(INCLUDED_DIR)

    print("=== head-EXCLUDED (§10) ===")
    print(summary_table(excluded).to_string(index=False))
    print("\n=== head-INCLUDED ===")
    print(summary_table(included).to_string(index=False))

    print("\n=== full head-included grid ===")
    print(included.to_string(index=False))

    print(f"\nhead_erank range (head-included): {included['head_erank'].min():.3f} - {included['head_erank'].max():.3f}")
    print(f"head_erank range (head-excluded): {excluded['head_erank'].min():.3f} - {excluded['head_erank'].max():.3f}")

    plot(excluded, included)
