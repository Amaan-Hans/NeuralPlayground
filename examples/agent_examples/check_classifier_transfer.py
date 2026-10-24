"""Does the has_value classifier key off a stable shape/identity cue rather
than anything shuffle-dependent?

Fits the has_value classifier on the FULL shuffle-enabled ("no LC") campaign
population (rm_real/rm_control/bl_real/bl_control, all 5 seeds - no held-out
split this time, since we're testing transfer to a genuinely different
dataset the classifier has never seen, not worried about leakage within
this one), then applies that FROZEN classifier to the no-shuffle ablation's
baseline population (bl_real/bl_control only - that campaign has no value
mechanism at all).

If the classifier is really reading an injected value signal, it should
call no-shuffle bl_real "not valuable" about as reliably as it does in its
own home campaign. If it's actually keying off something like distance-to-
reward or raw object identity (both present regardless of shuffling), it
should still flag no-shuffle bl_real as "valuable" at a similar - or even
higher - rate, since shuffling was the only thing disabled.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REWARD_PRED_SCRIPTS = r"C:\Users\hansl\Desktop\Projects\Masters_research\reward_prediction\scripts"
sys.path.insert(0, REWARD_PRED_SCRIPTS)
from common_decode import _drop_low_variance, get_feature_columns_by_scale  # noqa: E402

TRAIN_ROOT = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj"
TRAIN_SEEDS = [42, 7, 1, 2, 3]
TEST_ROOT = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle"
TEST_SEEDS = [42, 7, 2, 3]  # seed 1 (baseline) still training


def load_train_population():
    frames = []
    for seed in TRAIN_SEEDS:
        for condition in ["baseline", "reward_modulated"]:
            path = os.path.join(TRAIN_ROOT, f"seed_{seed}", condition, "allenvs_place_cells.csv")
            df = pd.read_csv(path)
            df["seed"] = seed
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    rm = df[df["condition"] == "reward_modulated"]
    bl = df[df["condition"] == "baseline"]
    rm_real = rm[(rm["is_landmark"] == 1) & rm["value"].notna()].copy()
    rm_control = rm[rm["is_control_landmark"] == 1].copy()
    bl_real = bl[bl["is_landmark"] == 1].copy()
    bl_control = bl[bl["is_control_landmark"] == 1].copy()
    rm_real["group"], rm_control["group"] = "rm_real", "rm_control"
    bl_real["group"], bl_control["group"] = "bl_real", "bl_control"
    pop = pd.concat([rm_real, rm_control, bl_real, bl_control], ignore_index=True)
    pop["has_value"] = (pop["group"] == "rm_real").astype(int)
    return pop


def load_test_population():
    frames = []
    for seed in TEST_SEEDS:
        path = os.path.join(TEST_ROOT, f"seed_{seed}", "baseline", "allenvs_place_cells.csv")
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    bl_real = df[df["is_landmark"] == 1].copy()
    bl_control = df[df["is_control_landmark"] == 1].copy()
    bl_real["group"], bl_control["group"] = "bl_real", "bl_control"
    return pd.concat([bl_real, bl_control], ignore_index=True)


def main():
    train_pop = load_train_population()
    test_pop = load_test_population()

    theta_cols = get_feature_columns_by_scale(train_pop)["Theta"]
    missing = [c for c in theta_cols if c not in test_pop.columns]
    if missing:
        print(f"WARNING: {len(missing)} theta columns missing in test population (e.g. {missing[:3]}) - "
              f"architectures may not match exactly.")
        theta_cols = [c for c in theta_cols if c in test_pop.columns]

    X_train = train_pop[theta_cols].to_numpy(dtype="float64")
    y_train = train_pop["has_value"].to_numpy()
    X_test = test_pop[theta_cols].to_numpy(dtype="float64")

    print(f"Fitting classifier on the FULL shuffle-enabled campaign population "
          f"(n={len(train_pop)}, {TRAIN_ROOT.split(os.sep)[-1]})...")
    X_train_f, X_test_f, keep = _drop_low_variance(X_train, X_test, verbose=False)
    scaler = StandardScaler().fit(X_train_f)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X_train_f), y_train)

    # Sanity check: the classifier's rate on its OWN home population (no held-out split - this is
    # the in-sample rate, expected to look similar to the earlier held-out pooled result).
    home_probs = clf.predict_proba(scaler.transform(X_train_f))[:, 1]
    home_preds = home_probs > 0.5
    print("\n=== in-sample rate on the training campaign itself (sanity check) ===")
    for g in ["rm_real", "rm_control", "bl_real", "bl_control"]:
        mask = (train_pop["group"] == g).to_numpy()
        print(f"  {g:12s} n={mask.sum():4d}  predicted positive rate={home_preds[mask].mean():.3f}")

    # The actual test: apply the FROZEN classifier to the no-shuffle baseline population it has
    # never seen at all (different campaign, different run, no value mechanism whatsoever).
    test_probs = clf.predict_proba(scaler.transform(X_test_f))[:, 1]
    test_preds = test_probs > 0.5
    print(f"\n=== transferred to the no-shuffle baseline population (n={len(test_pop)}, never seen in training) ===")
    for g in ["bl_real", "bl_control"]:
        mask = (test_pop["group"] == g).to_numpy()
        rate = test_preds[mask].mean() if mask.sum() else float("nan")
        prob_mean = test_probs[mask].mean() if mask.sum() else float("nan")
        print(f"  {g:12s} n={mask.sum():4d}  predicted positive rate={rate:.3f}  mean P(valuable)={prob_mean:.3f}")

    print("\n=== per-seed breakdown, no-shuffle bl_real ===")
    for seed in TEST_SEEDS:
        mask = ((test_pop["group"] == "bl_real") & (test_pop["seed"] == seed)).to_numpy()
        if mask.sum():
            print(f"  seed={seed}  n={mask.sum():4d}  rate={test_preds[mask].mean():.3f}")

    out = pd.DataFrame([
        {"population": "train_home", "group": g,
         "n": int((train_pop["group"] == g).sum()),
         "predicted_positive_rate": float(home_preds[(train_pop["group"] == g).to_numpy()].mean())}
        for g in ["rm_real", "rm_control", "bl_real", "bl_control"]
    ] + [
        {"population": "noshuffle_transfer", "group": g,
         "n": int((test_pop["group"] == g).sum()),
         "predicted_positive_rate": float(test_preds[(test_pop["group"] == g).to_numpy()].mean())}
        for g in ["bl_real", "bl_control"]
    ])
    out_path = os.path.join(os.path.dirname(__file__), "results_classifier_transfer.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
