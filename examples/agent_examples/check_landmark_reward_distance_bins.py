"""Follow-up to check_landmark_reward_predictiveness.py: which landmarks are
actually driving the distance-to-reward predictiveness - near ones only, or
do far-away landmarks also carry real information?

Note on scope: allenvs_place_cells.csv (the only data this uses - same as
the parent script, no new probing) has no raw object_id column, only an
is_landmark boolean - so "which landmark" can only be resolved by each
landmark's own actual distance-to-reward, not by its specific id 0-9. A
true per-id breakdown would need a fresh tem_probe_eval_allenvs_objectid.py
probe (it logs object_id) - not run here.

For each condition, pools the held-out (envs 12-15) predictions from
check_landmark_reward_predictiveness.py's per-seed Ridge models across all
5 seeds, splits them into near/mid/far TERTILES by actual distance-to-
-reward, and reports the predicted-vs-actual correlation *within* each
tertile. A significant positive correlation in the far tertile = far
landmarks still carry real, usable distance information, not just noise
that happens to average out near the model's global mean.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from check_landmark_reward_predictiveness import CAMPAIGNS, SEEDS, TRAIN_ENVS, TEST_ENVS, load_landmark_distance_df


def pooled_predictions(campaign_root, condition):
    all_actual, all_pred, all_seed = [], [], []
    for seed in SEEDS:
        path = os.path.join(campaign_root, f"seed_{seed}", condition, "allenvs_place_cells.csv")
        if not os.path.exists(path):
            continue
        df, feat_cols = load_landmark_distance_df(campaign_root, condition, seed)
        train = df[df["env_id"].isin(TRAIN_ENVS)]
        test = df[df["env_id"].isin(TEST_ENVS)]

        model = Ridge(alpha=1.0)
        model.fit(train[feat_cols].values, train["dist_to_reward"].values)
        pred = model.predict(test[feat_cols].values)

        all_actual.append(test["dist_to_reward"].values)
        all_pred.append(pred)
        all_seed.append(np.full(len(test), seed))

    return np.concatenate(all_actual), np.concatenate(all_pred), np.concatenate(all_seed)


def main():
    rows = []
    for label, (root, condition) in CAMPAIGNS.items():
        actual, pred, seeds = pooled_predictions(root, condition)
        print(f"\n=== {label} (pooled n={len(actual)} across {len(set(seeds))} seeds) ===")

        overall_r, overall_p = stats.pearsonr(actual, pred)
        print(f"  overall: r={overall_r:+.3f} (p={overall_p:.4f})  "
              f"actual_dist range=[{actual.min():.2f}, {actual.max():.2f}]")

        tertile_edges = np.quantile(actual, [1 / 3, 2 / 3])
        bin_names = ["near", "mid", "far"]
        bin_idx = np.digitize(actual, tertile_edges)  # 0=near,1=mid,2=far

        for b, name in enumerate(bin_names):
            mask = bin_idx == b
            if mask.sum() < 5:
                continue
            r, p = stats.pearsonr(actual[mask], pred[mask])
            mae = np.mean(np.abs(actual[mask] - pred[mask]))
            print(f"  {name:4s} tertile (n={mask.sum():3d}, actual {actual[mask].min():.2f}-{actual[mask].max():.2f}): "
                  f"r={r:+.3f} (p={p:.4f})  MAE={mae:.3f}  mean_actual={actual[mask].mean():.2f}  mean_pred={pred[mask].mean():.2f}")
            rows.append({
                "condition": label, "bin": name, "n": int(mask.sum()),
                "r": r, "p": p, "mae": mae,
                "mean_actual": actual[mask].mean(), "mean_pred": pred[mask].mean(),
            })

    out_path = os.path.join(os.path.dirname(__file__), "landmark_reward_distance_bins.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
