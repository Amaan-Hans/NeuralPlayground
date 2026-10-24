"""Does real injected value produce a decodable difference between
baseline and reward_modulated, holding position AND identity fixed by
construction - rather than hoping a classifier or a residualization
learns to ignore them?

baseline and reward_modulated share the EXACT SAME environment layout for
a given seed (object generation is seeded identically before the two
conditions diverge - verified directly: zero is_landmark/is_control_landmark
mismatches when matching rows by (seed, env_id, state_id)). So for any
landmark or control-landmark state, the object sitting there and its
physical position are identical in both conditions - only whether value
was actually injected differs.

Design: for each matched (seed, env_id, state_id), compute
diff = reward_modulated_features - baseline_features. Label diff=1 if
that state is a real (is_landmark) landmark, diff=0 if it's a control
landmark. Train a classifier to tell real-landmark diffs from
control-landmark diffs using ONLY the difference vectors. Position and
identity are held fixed on both sides of every single comparison - if
this is still decodable, that cannot be explained by either.
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

CAMPAIGNS = {
    "no LC": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj",
        "baseline", "reward_modulated",
    ),
    "LC gating": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_lc",
        "baseline_lc", "reward_modulated_lc",
    ),
    "no-shuffle ablation": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle",
        "baseline", "reward_modulated",
    ),
}
SEEDS = [42, 7, 1, 2, 3]
N_PERMUTATIONS = 300


def load_matched_diffs(root, bl_cond, rm_cond):
    frames = []
    theta_cols = None
    for seed in SEEDS:
        bl = pd.read_csv(os.path.join(root, f"seed_{seed}", bl_cond, "allenvs_place_cells.csv"))
        rm = pd.read_csv(os.path.join(root, f"seed_{seed}", rm_cond, "allenvs_place_cells.csv"))
        if theta_cols is None:
            theta_cols = get_feature_columns_by_scale(bl)["Theta"]

        bl_land = bl[(bl["is_landmark"] == 1) | (bl["is_control_landmark"] == 1)].copy()
        merged = bl_land.merge(
            rm, on=["env_id", "state_id"], suffixes=("_bl", "_rm")
        )
        # Sanity: the merge must match real-vs-control labels on both sides -
        # confirms layouts really are identical for this seed too.
        assert (merged["is_landmark_bl"] == merged["is_landmark_rm"]).all()
        assert (merged["is_control_landmark_bl"] == merged["is_control_landmark_rm"]).all()

        diff = merged[[c + "_rm" for c in theta_cols]].to_numpy() - merged[[c + "_bl" for c in theta_cols]].to_numpy()
        out = pd.DataFrame(diff, columns=theta_cols)
        out["seed"] = seed
        out["env_id"] = merged["env_id"]
        out["state_id"] = merged["state_id"]
        out["is_real"] = merged["is_landmark_bl"].astype(int)
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


def run_campaign(name, root, bl_cond, rm_cond):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    pop, theta_cols = load_matched_diffs(root, bl_cond, rm_cond)
    print(f"matched pairs: n={len(pop)}  real={pop['is_real'].sum()}  control={(pop['is_real'] == 0).sum()}")

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
    print(f"null mean={null_bal_acc.mean():.3f}  std={null_bal_acc.std():.3f}  p={p_value:.4g}")
    return {"campaign": name, "n": len(pop), "accuracy": acc, "balanced_accuracy": bal_acc,
            "auc": auc, "p_value": p_value}


def main():
    results = [run_campaign(name, *cfg) for name, cfg in CAMPAIGNS.items()]
    out_path = os.path.join(os.path.dirname(__file__), "results_matched_pair_value.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
