"""Properly controlled version of check_rm_lc_transfer_to_baseline.py's
idea: same has_value_full_training_test.py recipe as every other has_value
result in this project (positive = rm_lc's real landmarks; negative = rm_lc's
own controls + baseline's real landmarks + baseline's controls, ALL mixed
into ONE training population, group-safe held-out split, permutation test) -
but this time baseline is "baseline" (rotation ON, no LC), not "baseline_lc"
- i.e. deliberately pairing the value-trained LC network against a baseline
from across the LC axis too, not just across the reward axis.

This is the fix for check_rm_lc_transfer_to_baseline.py's confound: that
script trained only on rm_lc's own real-vs-control split, so the shape of
"landmark vs control" alone could (and did, at a 1.000 rate) explain the
transfer result. Mixing baseline in as negatives during training forces the
classifier to find something that separates value-bearing rows from
same-shaped non-value rows, not shape alone.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

REWARD_PRED_SCRIPTS = r"C:\Users\hansl\Desktop\Projects\Masters_research\reward_prediction\scripts"
sys.path.insert(0, REWARD_PRED_SCRIPTS)
from common_decode import SEED, _drop_low_variance, get_feature_columns_by_scale  # noqa: E402

THESIS = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis"
SEEDS = [42, 7, 1, 2, 3]
N_PERMUTATIONS = 300

RM_LC_ROOT = os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle_lc")
BASELINE_ROOT = os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle")


def load_condition(root, condition, tag):
    frames = []
    for seed in SEEDS:
        path = os.path.join(root, f"seed_{seed}", condition, "allenvs_place_cells.csv")
        df = pd.read_csv(path)
        df["seed"] = seed
        df["campaign_tag"] = tag  # keeps location_id unique even though both campaigns reuse seed/env_id/state_id numbering
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fit_eval(X_train, y_train, X_test, y_test):
    X_train_f, X_test_f, keep = _drop_low_variance(X_train, X_test, verbose=False)
    xs = StandardScaler().fit(X_train_f)
    X_train_s, X_test_s = xs.transform(X_train_f), xs.transform(X_test_f)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s)[:, 1]
    return preds, probs


def main():
    rm_lc = load_condition(RM_LC_ROOT, "reward_modulated_lc", "rm_lc")
    baseline = load_condition(BASELINE_ROOT, "baseline", "baseline")

    theta_cols = get_feature_columns_by_scale(rm_lc)["Theta"]

    rm_real = rm_lc[rm_lc["is_landmark"] == 1].copy()
    rm_control = rm_lc[rm_lc["is_control_landmark"] == 1].copy()
    bl_real = baseline[baseline["is_landmark"] == 1].copy()
    bl_control = baseline[baseline["is_control_landmark"] == 1].copy()

    rm_real["group"], rm_control["group"] = "rm_lc_real", "rm_lc_control"
    bl_real["group"], bl_control["group"] = "bl_real", "bl_control"

    pop = pd.concat([rm_real, rm_control, bl_real, bl_control], ignore_index=True)
    pop["has_value"] = (pop["group"] == "rm_lc_real").astype(int)
    print(pop.groupby(["group", "seed"]).size().unstack())

    X = pop[theta_cols].to_numpy(dtype="float64")
    y = pop["has_value"].to_numpy()
    location_id = (
        pop["campaign_tag"] + "_" + pop["seed"].astype(str) + "_" + pop["group"] + "_"
        + pop["env_id"].astype(str) + "_" + pop["state_id"].astype(str)
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(np.arange(len(pop)), y, groups=location_id))
    overlap = set(location_id.to_numpy()[train_idx]) & set(location_id.to_numpy()[test_idx])
    assert not overlap, f"{len(overlap)} locations leaked across the train/test split"

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = pop["group"].to_numpy()[test_idx]

    print(f"\ntrain n={len(train_idx)}  test (held out) n={len(test_idx)}")

    preds, probs = fit_eval(X_train, y_train, X_test, y_test)
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    print(f"\n=== held-out test: accuracy={acc:.3f}  balanced_accuracy={bal_acc:.3f}  AUC={auc:.3f} ===")
    print(pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]))

    print(f"\nrunning {N_PERMUTATIONS}-permutation null test on held-out balanced_accuracy...")
    rng = np.random.default_rng(SEED)
    null_bal_acc = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        y_train_perm = rng.permutation(y_train)
        preds_perm, _ = fit_eval(X_train, y_train_perm, X_test, y_test)
        null_bal_acc[i] = balanced_accuracy_score(y_test, preds_perm)
    p_value = (np.sum(null_bal_acc >= bal_acc) + 1) / (N_PERMUTATIONS + 1)
    print(f"true balanced_accuracy={bal_acc:.3f}  null mean={null_bal_acc.mean():.3f} "
          f"std={null_bal_acc.std():.3f}  p={p_value:.4g}")

    print("\n=== predicted has_value=1 rate by group (held-out test) ===")
    for g in ["rm_lc_real", "rm_lc_control", "bl_real", "bl_control"]:
        mask = groups_test == g
        rate = preds[mask].mean() if mask.sum() else float("nan")
        print(f"{g:14s} n={mask.sum():4d}  predicted positive rate={rate:.3f}  (true={'1' if g == 'rm_lc_real' else '0'})")


if __name__ == "__main__":
    main()
