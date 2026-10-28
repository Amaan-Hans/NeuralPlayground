"""Value decoding for the ring-loop experiment, directly from
probe_loop_ring_all.py's *_place_cells.csv files (no separate whole-room
pass needed - real landmarks, the real reward, and real controls all sit
on the same ring the agent actually walks). Same has_value_full_training_test.py
recipe as every other has_value result in this project: positive = that
pair's reward_modulated real landmarks, negative = that pair's own
reward_modulated controls + baseline real landmarks + baseline controls,
Theta-scale features, group-safe split (held out by seed/group/env_id/
ring_pos), 300-permutation significance test - run separately for "pre"
and "post" (the shift), for each of the 4 {rotation, LC} pairs.
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

LOOP_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop"
SEEDS = [42, 7, 1, 2, 3]
N_PERMUTATIONS = 300

PAIRS = {
    "rotation ON, no LC": ("baseline", "rm"),
    "rotation ON, LC": ("baseline_lc", "rm_lc"),
    "rotation OFF, no LC": ("baseline_noshuffle", "rm_noshuffle"),
    "rotation OFF, LC": ("baseline_lc_noshuffle", "rm_lc_noshuffle"),
}


def load_all(label):
    frames = []
    for seed in SEEDS:
        df = pd.read_csv(os.path.join(LOOP_DIR, f"{label}_seed{seed}_place_cells.csv"))
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fit_eval(X_train, y_train, X_test, y_test):
    X_train_f, X_test_f, keep = _drop_low_variance(X_train, X_test, verbose=False)
    xs = StandardScaler().fit(X_train_f)
    X_train_s, X_test_s = xs.transform(X_train_f), xs.transform(X_test_f)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_s, y_train)
    return clf.predict(X_test_s), clf.predict_proba(X_test_s)[:, 1]


def run_test(test_label, baseline_label, rm_label, half):
    bl_all = load_all(baseline_label)
    rm_all = load_all(rm_label)
    bl = bl_all[bl_all["half"] == half]
    rm = rm_all[rm_all["half"] == half]
    theta_cols = get_feature_columns_by_scale(bl)["Theta"]

    rm_real = rm[rm["is_landmark"] == 1].copy()
    rm_control = rm[rm["is_control_landmark"] == 1].copy()
    bl_real = bl[bl["is_landmark"] == 1].copy()
    bl_control = bl[bl["is_control_landmark"] == 1].copy()
    rm_real["group"], rm_control["group"] = "rm_real", "rm_control"
    bl_real["group"], bl_control["group"] = "bl_real", "bl_control"
    pop = pd.concat([rm_real, rm_control, bl_real, bl_control], ignore_index=True)
    pop["has_value"] = (pop["group"] == "rm_real").astype(int)

    X = pop[theta_cols].to_numpy(dtype="float64")
    y = pop["has_value"].to_numpy()
    location_id = (pop["seed"].astype(str) + "_" + pop["group"] + "_"
                   + pop["env_id"].astype(str) + "_" + pop["ring_pos"].astype(str))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(np.arange(len(pop)), y, groups=location_id))
    overlap = set(location_id.to_numpy()[train_idx]) & set(location_id.to_numpy()[test_idx])
    assert not overlap, f"{len(overlap)} locations leaked across the train/test split"

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = pop["group"].to_numpy()[test_idx]

    preds, probs = fit_eval(X_train, y_train, X_test, y_test)
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    rng = np.random.default_rng(SEED)
    null_bal_acc = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        preds_perm, _ = fit_eval(X_train, rng.permutation(y_train), X_test, y_test)
        null_bal_acc[i] = balanced_accuracy_score(y_test, preds_perm)
    p_value = (np.sum(null_bal_acc >= bal_acc) + 1) / (N_PERMUTATIONS + 1)

    group_rates = {}
    for g in ["rm_real", "rm_control", "bl_real", "bl_control"]:
        mask = groups_test == g
        group_rates[g] = preds[mask].mean() if mask.sum() else float("nan")

    print(f"[{test_label} / {half}] n_train={len(train_idx)} n_test={len(test_idx)}  "
          f"acc={acc:.3f} bal_acc={bal_acc:.3f} auc={auc:.3f} p={p_value:.4g}  "
          f"rm_real={group_rates['rm_real']:.3f} rm_control={group_rates['rm_control']:.3f} "
          f"bl_real={group_rates['bl_real']:.3f} bl_control={group_rates['bl_control']:.3f}")

    return {
        "test": test_label, "half": half, "accuracy": acc, "balanced_accuracy": bal_acc, "auc": auc,
        "permutation_p_value": p_value, "n_train": len(train_idx), "n_test": len(test_idx),
        **{f"rate_{g}": r for g, r in group_rates.items()},
    }


def main():
    results = []
    for test_label, (baseline_label, rm_label) in PAIRS.items():
        for half in ("pre", "post"):
            results.append(run_test(test_label, baseline_label, rm_label, half))
    out_df = pd.DataFrame(results)
    out_df.to_csv(os.path.join(LOOP_DIR, "value_decoding_pre_post.csv"), index=False)
    print(f"\nSaved {os.path.join(LOOP_DIR, 'value_decoding_pre_post.csv')}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
