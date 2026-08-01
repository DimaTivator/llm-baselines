"""Analyze the CoT-reasoning plasticity sweep (exps/plasticity_eval_cot/),
which tests arXiv:2602.11137's STRONGEST claim (Section 4.3, OLMo-2-1B-140x
example): a worse-pretrained model can beat a better-pretrained one on
ABSOLUTE post-fine-tune accuracy. Fine-tuned on MetaMathQA/SimpleScaling,
evaluated via greedy exact-match on GSM8KPlatinum+MATH subsamples.

Usage: python src/plasticity_cot_analyze.py
"""
import glob
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "exps" / "plasticity_eval_cot"


def load_results() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        d = json.load(open(f))
        size = "124M" if "llama124m" in f else ("257M" if "llama257m" in f else "500M")
        before, after = d["before"], d["after"]
        rows.append({
            "size": size,
            "ft_dataset": d["ft_dataset"],
            "coef": d["spectral_l1_reg_coef"],
            "gsm8k_before": before["gsm8k_platinum_acc"],
            "gsm8k_after": after["gsm8k_platinum_acc"],
            "gsm8k_delta": after["gsm8k_platinum_acc"] - before["gsm8k_platinum_acc"],
            "math_before": before["math_acc"],
            "math_after": after["math_acc"],
            "math_delta": after["math_acc"] - before["math_acc"],
            "combined_before": (before["gsm8k_platinum_acc"] + before["math_acc"]) / 2,
            "combined_after": (after["gsm8k_platinum_acc"] + after["math_acc"]) / 2,
        })
    return pd.DataFrame(rows).sort_values(["ft_dataset", "size", "coef"]).reset_index(drop=True)


def check_overtaking(df: pd.DataFrame):
    """For each (ft_dataset, size) group, check whether any higher-coef model's
    absolute after-accuracy beats the baseline's (coef=0) after-accuracy,
    despite the higher-coef model's before-accuracy being lower."""
    print("\n=== Overtaking check (worse pretraining -> better post-FT absolute accuracy?) ===")
    for (ft, size), g in df.groupby(["ft_dataset", "size"]):
        baseline = g[g["coef"] == 0]
        if baseline.empty:
            continue
        baseline = baseline.iloc[0]
        for _, row in g[g["coef"] != 0].iterrows():
            worse_pretrain = row["combined_before"] < baseline["combined_before"]
            better_postft = row["combined_after"] > baseline["combined_after"]
            flag = "*** OVERTAKE ***" if (worse_pretrain and better_postft) else ""
            print(f"{ft:14} {size:5} coef={row['coef']:<5} "
                  f"before={row['combined_before']:.4f} (baseline {baseline['combined_before']:.4f}) "
                  f"after={row['combined_after']:.4f} (baseline {baseline['combined_after']:.4f}) {flag}")


if __name__ == "__main__":
    df = load_results()
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    check_overtaking(df)
