"""Why does the has_value classifier sometimes flag bl_real (a baseline
landmark that was never given any value) as "valuable"? Two structural,
non-value differences exist between is_landmark==1 (bl_real - permanently
fixed per environment instance) and is_control_landmark==1 (bl_control -
reshuffled every control_shuffle_interval episodes):
  1. Placement bias: landmarks are drawn closer to reward_location than
     control landmarks (see DiscreteObjectEnvironment.generate_objects).
  2. Permanence: landmarks never move; controls do.

"Fixedness" is a constant within bl_real (every bl_real row is fixed) and
within bl_control (every row is reshuffled) - it cannot explain *within-
group* variation in the classifier's predicted probability. Distance to
reward CAN vary within bl_real. So: if misclassified (high predicted-
positive-probability) bl_real rows are the ones closer to reward_location,
that's evidence for the placement-bias confound specifically, not a vague
"fixed vs moving" story.

Runs the SAME classifier/train-test-split recipe as value_decode_probe.py
(must match exactly, or this isn't testing the same model), for both
campaigns, then for the bl_real test subset: correlates predicted
probability against distance-to-reward, and reports predicted-positive
rate for the near-half vs far-half (split at the median distance). Also
reports the structural fact on its own: mean distance-to-reward for
is_landmark==1 vs is_control_landmark==1, pooled across all seeds,
independent of any classifier.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

REWARD_PRED_SCRIPTS = r"C:\Users\hansl\Desktop\Projects\Masters_research\reward_prediction\scripts"
sys.path.insert(0, REWARD_PRED_SCRIPTS)
from common_decode import SEED, _drop_low_variance, get_feature_columns_by_scale  # noqa: E402

REWARD_LOCATION = (3.0, 3.0)
SEEDS = [42, 7, 1, 2, 3]

CAMPAIGNS = {
    "no LC": {
        "root": r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj",
        "baseline": "baseline",
        "reward_modulated": "reward_modulated",
    },
    "LC": {
        "root": r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_lc",
        "baseline": "baseline_lc",
        "reward_modulated": "reward_modulated_lc",
    },
}


def load_all(root, baseline_cond, rm_cond):
    frames = []
    for seed in SEEDS:
        for condition in [baseline_cond, rm_cond]:
            path = os.path.join(root, f"seed_{seed}", condition, "allenvs_place_cells.csv")
            df = pd.read_csv(path)
            df["seed"] = seed
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_population(df, baseline_cond, rm_cond):
    rm = df[df["condition"] == rm_cond]
    bl = df[df["condition"] == baseline_cond]

    rm_real = rm[(rm["is_landmark"] == 1) & rm["value"].notna()].copy()
    rm_control = rm[rm["is_control_landmark"] == 1].copy()
    bl_real = bl[bl["is_landmark"] == 1].copy()
    bl_control = bl[bl["is_control_landmark"] == 1].copy()

    rm_real["group"] = "rm_real"
    rm_control["group"] = "rm_control"
    bl_real["group"] = "bl_real"
    bl_control["group"] = "bl_control"

    pop = pd.concat([rm_real, rm_control, bl_real, bl_control], ignore_index=True)
    pop["has_value"] = (pop["group"] == "rm_real").astype(int)
    pop["dist_to_reward"] = np.sqrt(
        (pop["x"] - REWARD_LOCATION[0]) ** 2 + (pop["y"] - REWARD_LOCATION[1]) ** 2
    )
    return pop


def fit_eval(X_train, y_train, X_test):
    X_train_f, X_test_f, keep = _drop_low_variance(X_train, X_test, verbose=False)
    xs = StandardScaler().fit(X_train_f)
    X_train_s, X_test_s = xs.transform(X_train_f), xs.transform(X_test_f)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_s, y_train)
    probs = clf.predict_proba(X_test_s)[:, 1]
    return probs


def analyze_campaign(name, cfg):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    df = load_all(cfg["root"], cfg["baseline"], cfg["reward_modulated"])
    pop = build_population(df, cfg["baseline"], cfg["reward_modulated"])

    # --- Structural fact, independent of any classifier ---
    bl_real_dist = pop.loc[pop["group"] == "bl_real", "dist_to_reward"]
    bl_control_dist = pop.loc[pop["group"] == "bl_control", "dist_to_reward"]
    print(f"\n[structural] mean dist-to-reward: bl_real (fixed) = {bl_real_dist.mean():.2f} "
          f"(n={len(bl_real_dist)}), bl_control (reshuffled) = {bl_control_dist.mean():.2f} "
          f"(n={len(bl_control_dist)})")
    print(f"             i.e. landmarks sit {'closer to' if bl_real_dist.mean() < bl_control_dist.mean() else 'farther from'} "
          f"reward than controls, by construction (placement-bias confound candidate)")

    # --- Refit the exact same classifier/split as value_decode_probe.py ---
    theta_cols = get_feature_columns_by_scale(pop)["Theta"]
    X = pop[theta_cols].to_numpy(dtype="float64")
    y = pop["has_value"].to_numpy()
    location_id = (
        pop["seed"].astype(str) + "_" + pop["group"] + "_"
        + pop["env_id"].astype(str) + "_" + pop["state_id"].astype(str)
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(np.arange(len(pop)), y, groups=location_id))

    probs = fit_eval(X[train_idx], y[train_idx], X[test_idx])
    test_pop = pop.iloc[test_idx].copy()
    test_pop["pred_prob"] = probs

    bl_real_test = test_pop[test_pop["group"] == "bl_real"]
    print(f"\n[classifier] bl_real held-out test rows: n={len(bl_real_test)}")

    if len(bl_real_test) >= 8:
        r_pb, p_pb = pointbiserialr(bl_real_test["pred_prob"], bl_real_test["dist_to_reward"])
        r_sp, p_sp = spearmanr(bl_real_test["pred_prob"], bl_real_test["dist_to_reward"])
        print(f"  correlation(pred_prob, dist_to_reward): Pearson r={r_pb:.3f} (p={p_pb:.3g}), "
              f"Spearman rho={r_sp:.3f} (p={p_sp:.3g})")
        print("  (negative r = closer to reward -> higher predicted value-probability)")

        median_dist = bl_real_test["dist_to_reward"].median()
        near = bl_real_test[bl_real_test["dist_to_reward"] <= median_dist]
        far = bl_real_test[bl_real_test["dist_to_reward"] > median_dist]
        print(f"  near half (dist<={median_dist:.2f}, n={len(near)}): "
              f"mean pred_prob={near['pred_prob'].mean():.3f}, predicted-positive rate={(near['pred_prob'] > 0.5).mean():.3f}")
        print(f"  far half  (dist>{median_dist:.2f}, n={len(far)}): "
              f"mean pred_prob={far['pred_prob'].mean():.3f}, predicted-positive rate={(far['pred_prob'] > 0.5).mean():.3f}")
    else:
        print("  too few bl_real test rows for a reliable correlation")


def main():
    for name, cfg in CAMPAIGNS.items():
        analyze_campaign(name, cfg)


if __name__ == "__main__":
    main()
