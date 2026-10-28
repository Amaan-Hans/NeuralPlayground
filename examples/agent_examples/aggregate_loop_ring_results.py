"""Aggregate probe_loop_ring_all.py's 40 outputs:
  - pooled confusion matrices (summed across 5 seeds, pre vs post), raw +
    row-normalized, one pair per condition.
  - accuracy summary table (mean by category x half x condition).
"""
import os

import pandas as pd

LOOP_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop"
SEEDS = [42, 7, 1, 2, 3]
CONDITIONS = [
    "baseline", "rm", "baseline_lc", "rm_lc",
    "baseline_noshuffle", "rm_noshuffle", "baseline_lc_noshuffle", "rm_lc_noshuffle",
]
CATEGORIES = ["landmark", "control", "reward", "filler"]


def pooled_confusion(cond, half):
    total = pd.DataFrame(0, index=CATEGORIES, columns=CATEGORIES)
    for seed in SEEDS:
        cm = pd.read_csv(os.path.join(LOOP_DIR, f"{cond}_seed{seed}_confusion_{half}.csv"), index_col=0)
        total = total.add(cm, fill_value=0)
    return total.astype(int)


def accuracy_summary():
    rows = []
    for cond in CONDITIONS:
        frames = [pd.read_csv(os.path.join(LOOP_DIR, f"{cond}_seed{s}_episodes.csv")) for s in SEEDS]
        df = pd.concat(frames, ignore_index=True)
        for half in ("pre", "post"):
            sub = df[df["half"] == half]
            rows.append({
                "condition": cond, "half": half,
                **{f"acc_{c}": sub[f"acc_{c}"].mean() for c in CATEGORIES},
            })
    return pd.DataFrame(rows)


def main():
    acc_df = accuracy_summary()
    acc_df.to_csv(os.path.join(LOOP_DIR, "accuracy_summary.csv"), index=False)
    print(acc_df.to_string(index=False))

    for cond in CONDITIONS:
        for half in ("pre", "post"):
            cm = pooled_confusion(cond, half)
            cm.to_csv(os.path.join(LOOP_DIR, f"{cond}_confusion_{half}_pooled.csv"))
            row_pct = cm.div(cm.sum(axis=1), axis=0).round(3)
            row_pct.to_csv(os.path.join(LOOP_DIR, f"{cond}_confusion_{half}_pooled_rowpct.csv"))
    print(f"\nSaved pooled confusion matrices + accuracy_summary.csv to {LOOP_DIR}")


if __name__ == "__main__":
    main()
