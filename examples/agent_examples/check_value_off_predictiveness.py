"""Same held-out-by-environment distance-to-reward regression as
check_landmark_reward_predictiveness.py / check_control_landmark_reward_predictiveness.py,
but on the value-OFF probe CSVs from probe_value_off_allenvs.py (seed 42
only): trained weights kept as-is, value injection into x_c forced off at
probe time regardless of what the checkpoint was trained with.

Produces, for each of {landmark, control} x {baseline weights, rm weights}
(4 cells total - LC and non-LC reported together), the held-out R^2/r with
value silenced - directly comparable to the value-ON numbers already on
record, to isolate whether rm's advantage survives without the TD value
number being fed into the forward pass at all.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

CONDITIONS = ["baseline", "rm", "baseline_lc", "rm_lc"]
SEEDS = [42, 7, 1, 2, 3]
TRAIN_ENVS = list(range(12))
TEST_ENVS = list(range(12, 16))


def load_and_split(probe_dir, suffix, condition, seed, flag_col):
    path = os.path.join(probe_dir, f"{condition}_seed{seed}_{suffix}.csv")
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith(("p_Theta", "p_Beta", "p_Gamma", "p_High_Gamma", "p_Delta"))]

    reward_xy = df[df["is_reward"] == 1].set_index("env_id")[["x", "y"]]
    rows = df[df[flag_col] == 1].copy()
    rows["reward_x"] = rows["env_id"].map(reward_xy["x"])
    rows["reward_y"] = rows["env_id"].map(reward_xy["y"])
    rows["dist_to_reward"] = np.sqrt((rows["x"] - rows["reward_x"]) ** 2 + (rows["y"] - rows["reward_y"]) ** 2)

    train = rows[rows["env_id"].isin(TRAIN_ENVS)]
    test = rows[rows["env_id"].isin(TEST_ENVS)]
    return train[feat_cols].values, train["dist_to_reward"].values, test[feat_cols].values, test["dist_to_reward"].values


def run_one(probe_dir, suffix, condition, seed, flag_col):
    X_train, y_train, X_test, y_test = load_and_split(probe_dir, suffix, condition, seed, flag_col)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    ss_res = np.sum((y_test - pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    r, p = stats.pearsonr(y_test, pred)
    return r2, r, p, len(y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--suffix", required=True, help="e.g. valueoff or valueon")
    parser.add_argument("--label", required=True, help="e.g. 'value OFF' or 'value ON'")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for condition in CONDITIONS:
        for group, flag_col in [("landmark", "is_landmark"), ("control", "is_control_landmark")]:
            for seed in SEEDS:
                path = os.path.join(args.probe_dir, f"{condition}_seed{seed}_{args.suffix}.csv")
                if not os.path.exists(path):
                    print(f"missing: {path}")
                    continue
                r2, r, p, n_test = run_one(args.probe_dir, args.suffix, condition, seed, flag_col)
                rows.append({"condition": condition, "group": group, "seed": seed,
                             "r2": r2, "pearson_r": r, "p_value": p, "n_test": n_test})
                print(f"{condition:12s} {group:9s} seed={seed} ({args.label}): held-out R^2={r2:+.3f}  "
                      f"pearson_r={r:+.3f} (p={p:.4f})  n_test={n_test}")

    results = pd.DataFrame(rows)
    print(f"\n=== SUMMARY (mean +/- std across seeds), {args.label} ===")
    for condition in CONDITIONS:
        for group in ["landmark", "control"]:
            sub = results[(results["condition"] == condition) & (results["group"] == group)]
            if len(sub) == 0:
                continue
            print(f"{condition:12s} {group:9s} R^2={sub['r2'].mean():+.3f}+/-{sub['r2'].std():.3f}  "
                  f"pearson_r={sub['pearson_r'].mean():+.3f}+/-{sub['pearson_r'].std():.3f}  (n={len(sub)})")

    results.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
