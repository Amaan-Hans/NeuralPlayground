"""Aggregate probe_shuffle_midwalk.py's per-seed CSVs across all 5 seeds
for each of the 4 conditions, reporting mean +/- std for the before/drop/
after/final trend."""
import glob
import os

import numpy as np
import pandas as pd

OUT_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\NeuralPlayground\examples\agent_examples\shuffle_results"
CONDITIONS = ["baseline", "rm", "baseline_lc", "rm_lc"]
SEEDS = [42, 7, 1, 2, 3]


def main():
    rows = []
    for cond in CONDITIONS:
        befores, drops, afters, finals, drop_mags = [], [], [], [], []
        for seed in SEEDS:
            path = os.path.join(OUT_DIR, f"shuffle_{cond}_{seed}.csv")
            if not os.path.exists(path):
                print(f"missing: {path}")
                continue
            df = pd.read_csv(path)
            before = df.iloc[20:25]["x_gt"].mean()
            drop_chunk = df.iloc[25]["x_gt"]
            after5 = df.iloc[25:30]["x_gt"].mean()
            final5 = df.iloc[45:50]["x_gt"].mean()
            befores.append(before)
            drops.append(drop_chunk)
            afters.append(after5)
            finals.append(final5)
            drop_mags.append(before - drop_chunk)
            print(f"  {cond} seed={seed}: before={before:.3f} drop_chunk={drop_chunk:.3f} "
                  f"after5={after5:.3f} final5={final5:.3f} drop_mag={before - drop_chunk:.3f}")
        rows.append({
            "condition": cond,
            "before_mean": np.mean(befores), "before_std": np.std(befores),
            "drop_chunk_mean": np.mean(drops), "drop_chunk_std": np.std(drops),
            "after5_mean": np.mean(afters), "after5_std": np.std(afters),
            "final5_mean": np.mean(finals), "final5_std": np.std(finals),
            "drop_mag_mean": np.mean(drop_mags), "drop_mag_std": np.std(drop_mags),
            "n_seeds": len(befores),
        })

    summary = pd.DataFrame(rows)
    print("\n=== SUMMARY (mean +/- std across seeds) ===")
    for _, r in summary.iterrows():
        print(f"{r['condition']:14s} before={r['before_mean']:.3f}+/-{r['before_std']:.3f}  "
              f"drop_chunk={r['drop_chunk_mean']:.3f}+/-{r['drop_chunk_std']:.3f}  "
              f"drop_mag={r['drop_mag_mean']:.3f}+/-{r['drop_mag_std']:.3f}  "
              f"after5={r['after5_mean']:.3f}+/-{r['after5_std']:.3f}  "
              f"final5={r['final5_mean']:.3f}+/-{r['final5_std']:.3f}  (n={int(r['n_seeds'])})")

    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print(f"\nSaved {os.path.join(OUT_DIR, 'summary.csv')}")


if __name__ == "__main__":
    main()
