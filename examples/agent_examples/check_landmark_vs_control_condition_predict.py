"""Is cross-network divergence (baseline vs reward_modulated) bigger for
landmarks than for controls - using controls as the network-noise
baseline, rather than comparing against a totally different campaign
pairing (check_matched_pair_null.py)?

For each campaign, train a classifier to predict which CONDITION (baseline
or reward_modulated) produced a given row's raw place-cell activity -
once restricted to landmark rows only (is_landmark==1, real value-eligible
objects), once restricted to control rows only (is_control_landmark==1,
architecturally never eligible for value in either condition). Both tests
compare the exact same two trained networks; the only thing that differs
is which id-range is being read out, so this isolates whether value
produces EXTRA separability specifically for landmarks, beyond whatever
generic network-to-network noise affects both id-ranges equally.

If landmark-accuracy >> control-accuracy: value causes real, extra
divergence beyond generic noise.
If landmark-accuracy ~= control-accuracy: consistent with the null test's
finding that this is mostly/only generic cross-network noise, unrelated
to value.
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


def load_population(root, bl_cond, rm_cond):
    frames = []
    for seed in SEEDS:
        bl = pd.read_csv(os.path.join(root, f"seed_{seed}", bl_cond, "allenvs_place_cells.csv"))
        rm = pd.read_csv(os.path.join(root, f"seed_{seed}", rm_cond, "allenvs_place_cells.csv"))
        bl["seed"], rm["seed"] = seed, seed
        bl["is_rm"], rm["is_rm"] = 0, 1
        frames.append(bl)
        frames.append(rm)
    return pd.concat(frames, ignore_index=True)


def fit_eval(X_train, y_train, X_test):
    X_train_f, X_test_f, keep = _drop_low_variance(X_train, X_test, verbose=False)
    xs = StandardScaler().fit(X_train_f)
    X_train_s, X_test_s = xs.transform(X_train_f), xs.transform(X_test_f)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s)[:, 1]
    return preds, probs


def run_subset(pop, theta_cols, label):
    X = pop[theta_cols].to_numpy(dtype="float64")
    y = pop["is_rm"].to_numpy()
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

    rng = np.random.default_rng(SEED)
    null_bal_acc = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        y_train_perm = rng.permutation(y_train)
        preds_perm, _ = fit_eval(X_train, y_train_perm, X_test)
        null_bal_acc[i] = balanced_accuracy_score(y_test, preds_perm)
    p_value = (np.sum(null_bal_acc >= bal_acc) + 1) / (N_PERMUTATIONS + 1)

    print(f"  {label}: n={len(pop)} (train={len(train_idx)} test={len(test_idx)})  "
          f"accuracy={acc:.3f}  balanced_accuracy={bal_acc:.3f}  AUC={auc:.3f}  p={p_value:.4g}")
    return {"accuracy": acc, "balanced_accuracy": bal_acc, "auc": auc, "p_value": p_value, "n": len(pop)}


def main():
    results = []
    for name, (root, bl_cond, rm_cond) in CAMPAIGNS.items():
        print(f"\n{'=' * 70}\n{name}: predict condition (baseline vs reward_modulated) from raw activity\n{'=' * 70}")
        pop = load_population(root, bl_cond, rm_cond)
        theta_cols = get_feature_columns_by_scale(pop)["Theta"]

        landmark_pop = pop[pop["is_landmark"] == 1].copy()
        control_pop = pop[pop["is_control_landmark"] == 1].copy()

        r_land = run_subset(landmark_pop, theta_cols, "LANDMARKS only")
        r_ctrl = run_subset(control_pop, theta_cols, "CONTROLS only ")
        gap = r_land["balanced_accuracy"] - r_ctrl["balanced_accuracy"]
        print(f"  --> landmark - control balanced_accuracy gap: {gap:+.3f}")

        results.append({"campaign": name, "subset": "landmark", **r_land})
        results.append({"campaign": name, "subset": "control", **r_ctrl})

    out_path = os.path.join(os.path.dirname(__file__), "results_landmark_vs_control_condition_predict.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
