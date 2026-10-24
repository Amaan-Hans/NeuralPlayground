"""Does place-cell activity correlate with the actual learned value V(id),
using ONLY data from a single network and a single id-range - avoiding
both confounds found so far (cross-network divergence, and id-range
decodability)?

Restricted to rm_real rows only (is_landmark==1, value not NaN) - the
reward_modulated network's own landmarks, nothing else. No baseline rows,
no control-landmark rows, no cross-network comparison at all. Within each
(seed, env_id) environment instance, there are exactly n_landmarks=10
landmark ids, each with its own learned V(id) - these values differ
because TD(0) converges differently per id (mainly driven by how close
that landmark typically is to the reward). Rank-split each environment's
10 ids into "high value" (top half) and "low value" (bottom half) RELATIVE
TO EACH OTHER - this controls for any absolute-scale differences in V
across seeds/environments, testing only whether place-cell activity
tracks relative value within the same environment.

Held out by (seed, env_id) - an entire environment's 10 landmarks go to
either train or test, never split across both, so this tests genuine
generalization to new environments' value rankings, not memorization.
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
        "reward_modulated",
    ),
    "LC gating": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_lc",
        "reward_modulated_lc",
    ),
    "no-shuffle ablation": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle",
        "reward_modulated",
    ),
}
SEEDS = [42, 7, 1, 2, 3]
N_PERMUTATIONS = 300


def load_rm_real(root, rm_cond):
    frames = []
    for seed in SEEDS:
        df = pd.read_csv(os.path.join(root, f"seed_{seed}", rm_cond, "allenvs_place_cells.csv"))
        df = df[(df["is_landmark"] == 1) & df["value"].notna()].copy()
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def rank_split(pop):
    """Within each (seed, env_id) group of n_landmarks ids, label the
    top half of V as high_value=1, bottom half as high_value=0 - relative
    ranking only, controls for any absolute V-scale difference across
    environments/seeds."""
    def _label(group):
        median = group["value"].median()
        # Ties at the median go to 0 (arbitrary but consistent) so groups
        # split as evenly as possible; permutation test handles any residual
        # imbalance correctly regardless.
        return (group["value"] > median).astype(int)

    pop = pop.copy()
    pop["high_value"] = pop.groupby(["seed", "env_id"], group_keys=False).apply(_label)
    return pop


def fit_eval(X_train, y_train, X_test):
    X_train_f, X_test_f, keep = _drop_low_variance(X_train, X_test, verbose=False)
    xs = StandardScaler().fit(X_train_f)
    X_train_s, X_test_s = xs.transform(X_train_f), xs.transform(X_test_f)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s)[:, 1]
    return preds, probs


def run_campaign(name, root, rm_cond):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    pop = load_rm_real(root, rm_cond)
    pop = rank_split(pop)
    theta_cols = get_feature_columns_by_scale(pop)["Theta"]
    print(f"rm_real rows: n={len(pop)}  high_value=1: {pop['high_value'].sum()}  "
          f"high_value=0: {(pop['high_value'] == 0).sum()}")

    X = pop[theta_cols].to_numpy(dtype="float64")
    y = pop["high_value"].to_numpy()
    # Group by (seed, env_id) only - an entire environment's 10 landmarks
    # move together, testing generalization to unseen environments.
    group_id = pop["seed"].astype(str) + "_" + pop["env_id"].astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(np.arange(len(pop)), y, groups=group_id))
    overlap = set(group_id.to_numpy()[train_idx]) & set(group_id.to_numpy()[test_idx])
    assert not overlap, f"{len(overlap)} environments leaked across the split"

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"train n={len(train_idx)} ({len(set(group_id.to_numpy()[train_idx]))} envs)  "
          f"test n={len(test_idx)} ({len(set(group_id.to_numpy()[test_idx]))} envs)")

    preds, probs = fit_eval(X_train, y_train, X_test)
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    print(f"accuracy={acc:.3f}  balanced_accuracy={bal_acc:.3f}  AUC={auc:.3f}")
    print(pd.DataFrame(cm, index=["true_low", "true_high"], columns=["pred_low", "pred_high"]))

    rng = np.random.default_rng(SEED)
    null_bal_acc = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        y_train_perm = rng.permutation(y_train)
        preds_perm, _ = fit_eval(X_train, y_train_perm, X_test)
        null_bal_acc[i] = balanced_accuracy_score(y_test, preds_perm)
    p_value = (np.sum(null_bal_acc >= bal_acc) + 1) / (N_PERMUTATIONS + 1)
    print(f"permutation null mean={null_bal_acc.mean():.3f}  std={null_bal_acc.std():.3f}  p={p_value:.4g}")

    return {"campaign": name, "n": len(pop), "accuracy": acc, "balanced_accuracy": bal_acc,
            "auc": auc, "p_value": p_value}


def main():
    results = [run_campaign(name, root, rm_cond) for name, (root, rm_cond) in CAMPAIGNS.items()]
    out_path = os.path.join(os.path.dirname(__file__), "results_within_network_value_rank.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
