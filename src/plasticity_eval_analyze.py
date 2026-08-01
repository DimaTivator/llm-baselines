"""Analyze the plasticity sweep (exps/plasticity_eval/), which tests whether
arXiv:2602.11137's finding ("weight decay increases language model
plasticity") replicates for spectral (nuclear-norm) weight decay, using the
existing llama124m/257m checkpoints in exps/cf_bruteforce_124M/ (AdamWSpectralL1Reg,
spectral_l1_reg_coef swept). Fine-tuned on ARC-Easy; before/after downstream
accuracy also tracked on HellaSwag, ARC-Challenge, PIQA (held out, not
fine-tuned on) to check for forgetting/transfer.

Usage: python src/plasticity_eval_analyze.py
"""
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "exps" / "plasticity_eval"
OUT_PNG = REPO_ROOT / "spectral-wd-plasticity-arc-easy.png"


def load_results() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        d = json.load(open(f))
        size = "124M" if "llama124m" in f else "257M"
        before, after = d["before"], d["after"]
        rows.append({
            "size": size,
            "coef": d["spectral_l1_reg_coef"],
            "arc_easy_before": before["downstream/arc_easy/acc_v2"],
            "arc_easy_after": after["downstream/arc_easy/acc_v2"],
            "arc_easy_delta": after["downstream/arc_easy/acc_v2"] - before["downstream/arc_easy/acc_v2"],
            "hellaswag_before": before["downstream/hellaswag/len_norm_v2"],
            "hellaswag_after": after["downstream/hellaswag/len_norm_v2"],
            "arc_challenge_before": before["downstream/arc_challenge/len_norm_v2"],
            "arc_challenge_after": after["downstream/arc_challenge/len_norm_v2"],
            "piqa_before": before["downstream/piqa/len_norm_v2"],
            "piqa_after": after["downstream/piqa/len_norm_v2"],
        })
    return pd.DataFrame(rows).sort_values(["size", "coef"]).reset_index(drop=True)


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for size, g in df.groupby("size"):
        axes[0].plot(g["coef"], g["arc_easy_before"], marker="o", linestyle="--", label=f"{size} before FT")
        axes[0].plot(g["coef"], g["arc_easy_after"], marker="s", linestyle="-", label=f"{size} after FT")
        axes[1].plot(g["coef"], g["arc_easy_delta"] * 100, marker="o", label=f"{size} plasticity (delta pp)")
    axes[0].set_xlabel("spectral_l1_reg_coef (pretraining)")
    axes[0].set_ylabel("ARC-Easy accuracy")
    axes[0].set_title("ARC-Easy accuracy before/after fine-tuning")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("spectral_l1_reg_coef (pretraining)")
    axes[1].set_ylabel("ARC-Easy accuracy gain (pp)")
    axes[1].set_title("Plasticity: fine-tuning gain vs spectral WD coef")
    axes[1].legend(fontsize=8)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    df = load_results()
    print(df.to_string(index=False))
    plot(df)
