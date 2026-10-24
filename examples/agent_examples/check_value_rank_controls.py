"""Five control checks on check_within_network_value_rank.py's result
(0.81-0.85 balanced accuracy decoding relative learned value from rm_real
place-cell activity, held out by environment).

1. Predict the rank from object_id alone (one-hot). If this matches the
   place-cell classifier, the label itself is largely a deterministic
   function of id (e.g. systematic placement-order bias), not something
   place cells are specially encoding.
2. Predict the rank from raw distance-to-reward alone, and from
   distance+id. Place cells are position encoders by definition, so if
   distance alone explains the result, the finding is "value correlates
   with position" (expected, not interesting), not "value is specially
   represented."
3. Take BASELINE's place-cell features (zero value, ever) at the SAME
   matched states (layouts are identical across conditions for a given
   seed), label them with RM's rank labels. If baseline's features predict
   RM's labels above chance, the signal is explainable by id/position/
   generic structure alone - nothing specific to value being learned.
4. Residualize the LABEL itself: subtract each id's own mean value
   (pooled across all its occurrences) before rank-splitting within each
   environment, so a fixed "id X is just generally high-value" main
   effect can't carry the label. Run the classifier on this stricter label
   using RM features (expect signal if real) and baseline features
   (expect chance - sanity check the control itself is clean).
5. Leave-one-seed-out (5-fold): hold out an entire seed's network, not
   just some of its environments, for the ORIGINAL (non-residualized)
   within-network test - a stricter generalization check across
   independently-trained networks.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

REWARD_PRED_SCRIPTS = r"C:\Users\hansl\Desktop\Projects\Masters_research\reward_prediction\scripts"
sys.path.insert(0, REWARD_PRED_SCRIPTS)
from common_decode import SEED, _drop_low_variance, get_feature_columns_by_scale  # noqa: E402

N_PERMUTATIONS = 300
REWARD_LOCATION = (3.0, 3.0)


def load_pair(probe_dir, seeds):
    frames = []
    for seed in seeds:
        df = pd.read_csv(os.path.join(probe_dir, f"objectid_seed{seed}.csv"))
        bl = df[df["condition"] == "baseline"]
        rm = df[df["condition"] == "reward_modulated"]
        rm_real = rm[(rm["is_landmark"] == 1) & rm["value"].notna()].copy()
        merged = rm_real.merge(bl, on=["env_id", "state_id"], suffixes=("_rm", "_bl"))
        assert (merged["is_landmark_rm"] == merged["is_landmark_bl"]).all()
        assert (merged["object_id_rm"] == merged["object_id_bl"]).all()
        merged["object_id"] = merged["object_id_rm"]
        merged["seed"] = seed
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def add_labels(pop):
    pop = pop.copy()
    pop["dist_to_reward"] = np.sqrt(
        (pop["x_rm"] - REWARD_LOCATION[0]) ** 2 + (pop["y_rm"] - REWARD_LOCATION[1]) ** 2
    )
    # Label A: plain within-environment rank split (the original test).
    pop["high_value"] = pop.groupby(["seed", "env_id"], group_keys=False)["value_rm"].apply(
        lambda s: (s > s.median()).astype(int)
    )
    # Label B: residualize value by object_id's own mean first (removes any
    # fixed "this id is just generally high/low value" main effect), THEN
    # rank-split within environment on the residual.
    id_mean = pop.groupby("object_id")["value_rm"].transform("mean")
    pop["value_resid"] = pop["value_rm"] - id_mean
    pop["high_value_resid"] = pop.groupby(["seed", "env_id"], group_keys=False)["value_resid"].apply(
        lambda s: (s > s.median()).astype(int)
    )
    return pop


def fit_eval(X_train, y_train, X_test):
    X_train_f, X_test_f, keep = _drop_low_variance(X_train, X_test, verbose=False)
    xs = StandardScaler().fit(X_train_f)
    X_train_s, X_test_s = xs.transform(X_train_f), xs.transform(X_test_f)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s)[:, 1]
    return preds, probs


def run_check(label, X, y, groups, permute=True):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(np.arange(len(X)), y, groups=groups))
    overlap = set(groups.to_numpy()[train_idx]) & set(groups.to_numpy()[test_idx])
    assert not overlap

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    preds, probs = fit_eval(X_train, y_train, X_test)
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = float("nan")

    p_value = None
    if permute:
        rng = np.random.default_rng(SEED)
        null_bal_acc = np.empty(N_PERMUTATIONS)
        for i in range(N_PERMUTATIONS):
            y_train_perm = rng.permutation(y_train)
            preds_perm, _ = fit_eval(X_train, y_train_perm, X_test)
            null_bal_acc[i] = balanced_accuracy_score(y_test, preds_perm)
        p_value = (np.sum(null_bal_acc >= bal_acc) + 1) / (N_PERMUTATIONS + 1)

    p_str = f"  p={p_value:.4g}" if p_value is not None else ""
    print(f"  {label}: n_train={len(train_idx)} n_test={len(test_idx)}  "
          f"accuracy={acc:.3f}  balanced_accuracy={bal_acc:.3f}  AUC={auc:.3f}{p_str}")
    return {"check": label, "accuracy": acc, "balanced_accuracy": bal_acc, "auc": auc, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", required=True,
                         help="Folder containing objectid_seed{N}.csv files for this campaign.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                         help="Seeds to pool (must have both baseline and reward_modulated rows).")
    parser.add_argument("--label", default=None, help="Campaign name for the output filename/printout.")
    args = parser.parse_args()
    campaign_label = args.label or os.path.basename(os.path.dirname(args.probe_dir.rstrip("\\/")))
    print(f"Campaign: {campaign_label}  seeds={args.seeds}  probe_dir={args.probe_dir}")

    # Compute base theta column names from an unsuffixed, pre-merge CSV -
    # after merging rm/bl (suffixes _rm/_bl), get_feature_columns_by_scale
    # would match BOTH suffixed variants at once (both start with "p_Theta"),
    # so it must run before the merge, not after.
    sample = pd.read_csv(os.path.join(args.probe_dir, f"objectid_seed{args.seeds[0]}.csv"))
    theta_cols = get_feature_columns_by_scale(sample)["Theta"]
    theta_cols_rm = [c + "_rm" for c in theta_cols]
    theta_cols_bl = [c + "_bl" for c in theta_cols]

    pop = load_pair(args.probe_dir, args.seeds)
    pop = add_labels(pop)

    env_group = pop["seed"].astype(str) + "_" + pop["env_id"].astype(str)
    results = []

    print("\n=== Check 1: id-alone (one-hot object_id, no place-cell features) ===")
    id_onehot = pd.get_dummies(pop["object_id"], prefix="id").to_numpy(dtype="float64")
    results.append(run_check("id-only -> high_value", id_onehot, pop["high_value"].to_numpy(), env_group))

    print("\n=== Check 2: distance-to-reward alone, and distance+id ===")
    dist_only = pop[["dist_to_reward"]].to_numpy(dtype="float64")
    results.append(run_check("distance-only -> high_value", dist_only, pop["high_value"].to_numpy(), env_group))
    dist_id = np.concatenate([dist_only, id_onehot], axis=1)
    results.append(run_check("distance+id -> high_value", dist_id, pop["high_value"].to_numpy(), env_group))

    print("\n=== Check 3: BASELINE's own place-cell features, labeled with RM's rank ===")
    X_bl = pop[theta_cols_bl].to_numpy(dtype="float64")
    results.append(run_check("baseline features -> RM's high_value label", X_bl, pop["high_value"].to_numpy(), env_group))

    print("\n=== Check 4: id-residualized label (removes each id's own mean-value main effect) ===")
    X_rm = pop[theta_cols_rm].to_numpy(dtype="float64")
    results.append(run_check("RM features -> residualized label", X_rm, pop["high_value_resid"].to_numpy(), env_group))
    results.append(run_check("baseline features -> residualized label (sanity null)",
                              X_bl, pop["high_value_resid"].to_numpy(), env_group))

    print("\n=== Check 5: leave-one-seed-out (original label, RM features) ===")
    fold_results = []
    for held_seed in args.seeds:
        train_mask = (pop["seed"] != held_seed).to_numpy()
        test_mask = (pop["seed"] == held_seed).to_numpy()
        X_train, X_test = X_rm[train_mask], X_rm[test_mask]
        y_train, y_test = pop["high_value"].to_numpy()[train_mask], pop["high_value"].to_numpy()[test_mask]
        preds, probs = fit_eval(X_train, y_train, X_test)
        bal_acc = balanced_accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        print(f"  held-out seed={held_seed}: n_test={len(y_test)}  balanced_accuracy={bal_acc:.3f}  AUC={auc:.3f}")
        fold_results.append(bal_acc)
    mean_loso = float(np.mean(fold_results))
    print(f"  mean leave-one-seed-out balanced_accuracy: {mean_loso:.3f}")
    results.append({"check": "leave-one-seed-out (mean)", "accuracy": None, "balanced_accuracy": mean_loso,
                     "auc": None, "p_value": None})

    out_path = os.path.join(os.path.dirname(__file__), f"results_value_rank_controls_{campaign_label}.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
