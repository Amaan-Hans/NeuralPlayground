"""Speculative test, flagged by the user as likely confounded before running
it: train a has_value classifier SOLELY on rm_lc's own real-vs-control
landmark split (rotation ON, LC) - no baseline mixed in as a negative this
time, unlike every other has_value test in this project - then transfer
(apply, no refitting) that classifier to plain baseline (rotation ON, no
LC)'s real and control landmarks, a network it never saw during training at
all.

This is deliberately the OLD, confound-prone design (documented in
has_value_full_training_test.py's own docstring as producing a 0.85 false-
positive transfer rate explained by a shape confound: is_landmark vs
is_control_landmark are trivially separable by shape/placement-bias alone
within a single network, and that separator transfers to any other
network's same-shape landmarks regardless of value). Included here anyway,
at the user's request, specifically to see what number it produces for the
rm_lc -> baseline pairing, with that caveat already on record before
looking at the result.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REWARD_PRED_SCRIPTS = r"C:\Users\hansl\Desktop\Projects\Masters_research\reward_prediction\scripts"
sys.path.insert(0, REWARD_PRED_SCRIPTS)
from common_decode import get_feature_columns_by_scale  # noqa: E402

THESIS = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis"
SEEDS = [42, 7, 1, 2, 3]

RM_LC_ROOT = os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_lc")
BASELINE_ROOT = os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj")


def load_condition(root, condition):
    frames = []
    for seed in SEEDS:
        path = os.path.join(root, f"seed_{seed}", condition, "allenvs_place_cells.csv")
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    rm_lc = load_condition(RM_LC_ROOT, "reward_modulated_lc")
    baseline = load_condition(BASELINE_ROOT, "baseline")

    theta_cols = get_feature_columns_by_scale(rm_lc)["Theta"]

    # Train population: rm_lc ONLY, real vs control - no baseline mixed in.
    rm_real = rm_lc[rm_lc["is_landmark"] == 1].copy()
    rm_control = rm_lc[rm_lc["is_control_landmark"] == 1].copy()
    rm_real["group"], rm_control["group"] = "rm_lc_real", "rm_lc_control"
    train_pop = pd.concat([rm_real, rm_control], ignore_index=True)
    train_pop["has_value"] = (train_pop["group"] == "rm_lc_real").astype(int)

    X_train = train_pop[theta_cols].to_numpy(dtype="float64")
    y_train = train_pop["has_value"].to_numpy()
    print(f"train population (rm_lc only): n={len(train_pop)}  "
          f"rm_lc_real={len(rm_real)}  rm_lc_control={len(rm_control)}")

    stds = X_train.std(axis=0)
    keep = stds > 1e-6
    print(f"  (dropped {int((~keep).sum())}/{len(keep)} near-constant Theta features, fit on rm_lc train data)")
    X_train_f = X_train[:, keep]

    xs = StandardScaler().fit(X_train_f)
    X_train_s = xs.transform(X_train_f)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_s, y_train)

    # Sanity check: how well does it do on ITS OWN population (in-sample, no held-out split -
    # this is not the headline number, just confirms the classifier learned *something* real/control-separating).
    self_preds = clf.predict(X_train_s)
    self_acc = (self_preds == y_train).mean()
    print(f"in-sample accuracy on rm_lc's own real-vs-control split: {self_acc:.3f}")

    # Transfer population: baseline (rotation ON, no LC) - never seen during training.
    bl_real = baseline[baseline["is_landmark"] == 1].copy()
    bl_control = baseline[baseline["is_control_landmark"] == 1].copy()

    for label, sub in [("baseline real landmarks (bl_real)", bl_real),
                        ("baseline control landmarks (bl_control)", bl_control)]:
        X = sub[theta_cols].to_numpy(dtype="float64")[:, keep]
        X_s = xs.transform(X)
        preds = clf.predict(X_s)
        probs = clf.predict_proba(X_s)[:, 1]
        print(f"{label}: n={len(sub)}  predicted-positive rate={preds.mean():.3f}  "
              f"mean P(has_value)={probs.mean():.3f}")


if __name__ == "__main__":
    main()
