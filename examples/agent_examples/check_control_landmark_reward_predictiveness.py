"""Negative-control counterpart to check_landmark_reward_predictiveness.py:
same held-out-by-environment Ridge regression (place-cell activity -> this
object's distance to the reward state), but run on CONTROL landmarks
(is_control_landmark==1) instead of value landmarks (is_landmark==1).

Controls are never value-bearing and get periodically reshuffled during
training (control_shuffle_interval=25 episodes) - the allenvs_place_cells.csv
snapshot here is just their position at the final episode (50000), same as
for value landmarks, so this is an as-fair-as-possible apples-to-apples
comparison: same regression, same features, same held-out split, same
checkpoints, only the object class differs. Expectation per the user: should
be LESS predictive of distance-to-reward than real landmarks, since nothing
about training ties a control landmark's identity or position to reward.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from check_landmark_reward_predictiveness import CAMPAIGNS, SEEDS, TRAIN_ENVS, TEST_ENVS, load_landmark_distance_df

FLAG_COL = "is_control_landmark"


def run_one(campaign_root, condition, seed):
    df, feat_cols = load_landmark_distance_df(campaign_root, condition, seed, flag_col=FLAG_COL)

    train = df[df["env_id"].isin(TRAIN_ENVS)]
    test = df[df["env_id"].isin(TEST_ENVS)]

    X_train, y_train = train[feat_cols].values, train["dist_to_reward"].values
    X_test, y_test = test[feat_cols].values, test["dist_to_reward"].values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    ss_res = np.sum((y_test - pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    r, p = stats.pearsonr(y_test, pred)
    return r2, r, p, len(test)


def main():
    rows = []
    for label, (root, condition) in CAMPAIGNS.items():
        for seed in SEEDS:
            path = os.path.join(root, f"seed_{seed}", condition, "allenvs_place_cells.csv")
            if not os.path.exists(path):
                print(f"missing: {path}")
                continue
            r2, r, p, n_test = run_one(root, condition, seed)
            rows.append({"condition": label, "seed": seed, "r2": r2, "pearson_r": r, "p_value": p, "n_test": n_test})
            print(f"{label:12s} seed={seed}: held-out R^2={r2:+.3f}  pearson_r={r:+.3f} (p={p:.4f})  n_test={n_test}")

    results = pd.DataFrame(rows)
    print("\n=== SUMMARY (mean +/- std across seeds), CONTROL landmarks ===")
    for label in CAMPAIGNS:
        sub = results[results["condition"] == label]
        if len(sub) == 0:
            continue
        print(f"{label:12s} R^2={sub['r2'].mean():+.3f}+/-{sub['r2'].std():.3f}  "
              f"pearson_r={sub['pearson_r'].mean():+.3f}+/-{sub['pearson_r'].std():.3f}  (n={len(sub)})")

    out_path = os.path.join(os.path.dirname(__file__), "control_landmark_reward_predictiveness.csv")
    results.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
