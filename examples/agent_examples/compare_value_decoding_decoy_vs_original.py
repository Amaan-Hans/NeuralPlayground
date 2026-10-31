"""Compares value_decoding_pre_post.csv between the original campaign
(Thesis/quantitative/loop) and the decoy-tiled campaign (Thesis2/quantitative/loop).
No rerun - both files already exist. Merges on (test, half), adds _orig/_decoy/_delta
columns for accuracy, balanced_accuracy, auc, rate_rm_real, rate_bl_real.
"""
import os

import pandas as pd

ORIG_PATH = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop\value_decoding_pre_post.csv"
DECOY_PATH = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis2\quantitative\loop\value_decoding_pre_post.csv"
OUT_PATH = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis2\quantitative\loop\value_decoding_compare_decoy_vs_original.csv"

METRICS = ["accuracy", "balanced_accuracy", "auc", "rate_rm_real", "rate_bl_real"]


def main():
    orig = pd.read_csv(ORIG_PATH)
    decoy = pd.read_csv(DECOY_PATH)
    merged = orig.merge(decoy, on=["test", "half"], suffixes=("_orig", "_decoy"))
    for m in METRICS:
        merged[f"{m}_delta"] = merged[f"{m}_decoy"] - merged[f"{m}_orig"]
    cols = ["test", "half"] + [f"{m}{suf}" for m in METRICS for suf in ("_orig", "_decoy", "_delta")]
    merged = merged[cols]
    merged.to_csv(OUT_PATH, index=False)
    for m in METRICS:
        print(f"\n=== {m} ===")
        print(merged[["test", "half", f"{m}_orig", f"{m}_decoy", f"{m}_delta"]].to_string(index=False))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
