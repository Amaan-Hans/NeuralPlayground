"""How predictive is place-cell activity AT landmark states of distance to
the reward object, and does this differ across the 4 conditions (baseline,
reward_modulated, baseline_lc, reward_modulated_lc)?

Uses the allenvs_place_cells.csv already bundled with every checkpoint (one
row per state per env, logged at the final training episode) - no new
probing or GPU compute needed. For each env_id, the single is_reward==1 row
gives that env's reward-state (x, y); every is_landmark==1 row's Euclidean
distance to that point is the regression target. Features are all place-cell
columns (p_Theta/p_Beta/p_Gamma/p_High_Gamma/p_Delta, ~440 dims).

Held out by environment (train envs 0-11, test envs 12-15, matching this
project's standing convention for avoiding within-env leakage) - Ridge
regression, report held-out R² and Pearson r on the test landmarks, per
seed, then mean+/-std across seeds per condition.

A high R² means: knowing the network's place-cell code at a landmark lets
you recover how far that landmark is from reward reasonably well. Compared
across conditions, it answers "does training with value (or value+LC) make
landmark activity carry more information about reward-proximity than a
baseline network with the same landmarks does?"
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

CAMPAIGNS = {
    "baseline": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj",
        "baseline",
    ),
    "rm": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj",
        "reward_modulated",
    ),
    "baseline_lc": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_lc",
        "baseline_lc",
    ),
    "rm_lc": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_lc",
        "reward_modulated_lc",
    ),
}
SEEDS = [42, 7, 1, 2, 3]
TRAIN_ENVS = list(range(12))
TEST_ENVS = list(range(12, 16))


def load_landmark_distance_df(campaign_root, condition, seed, flag_col="is_landmark"):
    path = os.path.join(campaign_root, f"seed_{seed}", condition, "allenvs_place_cells.csv")
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith(("p_Theta", "p_Beta", "p_Gamma", "p_High_Gamma", "p_Delta"))]

    reward_xy = df[df["is_reward"] == 1].set_index("env_id")[["x", "y"]]
    landmarks = df[df[flag_col] == 1].copy()
    landmarks["reward_x"] = landmarks["env_id"].map(reward_xy["x"])
    landmarks["reward_y"] = landmarks["env_id"].map(reward_xy["y"])
    landmarks["dist_to_reward"] = np.sqrt(
        (landmarks["x"] - landmarks["reward_x"]) ** 2 + (landmarks["y"] - landmarks["reward_y"]) ** 2
    )
    return landmarks, feat_cols


def run_one(campaign_root, condition, seed, flag_col="is_landmark"):
    df, feat_cols = load_landmark_distance_df(campaign_root, condition, seed, flag_col)

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
    print("\n=== SUMMARY (mean +/- std across seeds) ===")
    for label in CAMPAIGNS:
        sub = results[results["condition"] == label]
        if len(sub) == 0:
            continue
        print(f"{label:12s} R^2={sub['r2'].mean():+.3f}+/-{sub['r2'].std():.3f}  "
              f"pearson_r={sub['pearson_r'].mean():+.3f}+/-{sub['pearson_r'].std():.3f}  (n={len(sub)})")

    out_path = os.path.join(os.path.dirname(__file__), "landmark_reward_predictiveness.csv")
    results.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
