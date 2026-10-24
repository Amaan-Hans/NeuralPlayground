"""Null-hypothesis control for check_matched_pair_value.py, per a sharp
critique: that test diffed two SEPARATELY TRAINED networks
(reward_modulated - baseline), holding position and the object's identity
fixed. But it did not hold the fact that the two sides are different
trained functions fixed - f_rm(id) - f_bl(id) can differ systematically by
id for reasons having nothing to do with value, and since real landmarks
are always ids 0-9 and controls always 10-19, any id-correlated network
divergence could masquerade as a value signal.

This runs the identical real-vs-control classifier on a pairing where
NEITHER side ever carries value: no-LC campaign's baseline vs LC
campaign's baseline_lc, same seed. Layout match verified separately
(zero is_landmark/is_control_landmark mismatches, all 5 seeds) - these
two conditions share the exact same environment layout per seed.

If this scores near chance: the original matched-pair result is real
evidence of value, not just network-to-network noise.
If it also scores ~80%: the original test was mostly measuring generic
differences between separately-trained networks, not value - LC gating
is itself a real, systematic intervention though, so a high score here
is not fully conclusive either (see caveat at the end).
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

NOLC_ROOT = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj"
LC_ROOT = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_lc"
SEEDS = [42, 7, 1, 2, 3]
N_PERMUTATIONS = 300


def load_matched_diffs():
    frames = []
    theta_cols = None
    for seed in SEEDS:
        a = pd.read_csv(os.path.join(NOLC_ROOT, f"seed_{seed}", "baseline", "allenvs_place_cells.csv"))
        b = pd.read_csv(os.path.join(LC_ROOT, f"seed_{seed}", "baseline_lc", "allenvs_place_cells.csv"))
        if theta_cols is None:
            theta_cols = get_feature_columns_by_scale(a)["Theta"]

        a_land = a[(a["is_landmark"] == 1) | (a["is_control_landmark"] == 1)].copy()
        merged = a_land.merge(b, on=["env_id", "state_id"], suffixes=("_a", "_b"))
        assert (merged["is_landmark_a"] == merged["is_landmark_b"]).all()
        assert (merged["is_control_landmark_a"] == merged["is_control_landmark_b"]).all()

        diff = merged[[c + "_b" for c in theta_cols]].to_numpy() - merged[[c + "_a" for c in theta_cols]].to_numpy()
        out = pd.DataFrame(diff, columns=theta_cols)
        out["seed"] = seed
        out["env_id"] = merged["env_id"]
        out["state_id"] = merged["state_id"]
        out["is_real"] = merged["is_landmark_a"].astype(int)
        frames.append(out)
    return pd.concat(frames, ignore_index=True), theta_cols


def fit_eval(X_train, y_train, X_test):
    X_train_f, X_test_f, keep = _drop_low_variance(X_train, X_test, verbose=False)
    xs = StandardScaler().fit(X_train_f)
    X_train_s, X_test_s = xs.transform(X_train_f), xs.transform(X_test_f)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s)[:, 1]
    return preds, probs


def main():
    pop, theta_cols = load_matched_diffs()
    print(f"matched pairs (zero value on BOTH sides): n={len(pop)}  "
          f"real={pop['is_real'].sum()}  control={(pop['is_real'] == 0).sum()}")

    X = pop[theta_cols].to_numpy(dtype="float64")
    y = pop["is_real"].to_numpy()
    location_id = pop["seed"].astype(str) + "_" + pop["env_id"].astype(str) + "_" + pop["state_id"].astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(np.arange(len(pop)), y, groups=location_id))
    overlap = set(location_id.to_numpy()[train_idx]) & set(location_id.to_numpy()[test_idx])
    assert not overlap

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    preds, probs = fit_eval(X_train, y_train, X_test)
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    print(f"train n={len(train_idx)}  test n={len(test_idx)}")
    print(f"accuracy={acc:.3f}  balanced_accuracy={bal_acc:.3f}  AUC={auc:.3f}")
    print(pd.DataFrame(cm, index=["true_control", "true_real"], columns=["pred_control", "pred_real"]))

    rng = np.random.default_rng(SEED)
    null_bal_acc = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        y_train_perm = rng.permutation(y_train)
        preds_perm, _ = fit_eval(X_train, y_train_perm, X_test)
        null_bal_acc[i] = balanced_accuracy_score(y_test, preds_perm)
    p_value = (np.sum(null_bal_acc >= bal_acc) + 1) / (N_PERMUTATIONS + 1)
    print(f"permutation null mean={null_bal_acc.mean():.3f}  std={null_bal_acc.std():.3f}  p={p_value:.4g}")

    out_path = os.path.join(os.path.dirname(__file__), "results_matched_pair_null.csv")
    pd.DataFrame([{"accuracy": acc, "balanced_accuracy": bal_acc, "auc": auc, "p_value": p_value}]).to_csv(
        out_path, index=False
    )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
