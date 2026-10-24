"""The real thesis question, reframed: value injected into rm_real is
decodable almost by construction (it's written directly into the
compressed sensory code, flowing through the same pipeline as any other
sensory dimension) - that's not an interesting finding on its own. The
interesting question is whether a run with ZERO injected value anywhere
still has its permanently-fixed objects (landmarks) mistaken for
"valuable" by the same classifier - and whether that holds up whether
measured from the live training snapshot or an independent frozen-weight
probe, and whether shuffling (landmarks fixed forever vs controls
periodically reshuffled) makes this worse or better than no shuffling.

Runs the SAME has_value classifier recipe on a campaign's STANDALONE
PROBE data (frozen weights, independently walked - not the live training
snapshot), for direct comparison against the already-known training-
snapshot numbers.
"""
import argparse
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

N_PERMUTATIONS = 300


def load_all(probe_dir, seeds):
    frames = []
    for seed in seeds:
        df = pd.read_csv(os.path.join(probe_dir, f"objectid_seed{seed}.csv"))
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_population(df, bl_cond, rm_cond):
    rm = df[df["condition"] == rm_cond]
    bl = df[df["condition"] == bl_cond]
    rm_real = rm[(rm["is_landmark"] == 1) & rm["value"].notna()].copy()
    rm_control = rm[rm["is_control_landmark"] == 1].copy()
    bl_real = bl[bl["is_landmark"] == 1].copy()
    bl_control = bl[bl["is_control_landmark"] == 1].copy()
    rm_real["group"], rm_control["group"] = "rm_real", "rm_control"
    bl_real["group"], bl_control["group"] = "bl_real", "bl_control"
    pop = pd.concat([rm_real, rm_control, bl_real, bl_control], ignore_index=True)
    pop["has_value"] = (pop["group"] == "rm_real").astype(int)
    return pop


def fit_eval(X_train, y_train, X_test):
    X_train_f, X_test_f, keep = _drop_low_variance(X_train, X_test, verbose=False)
    xs = StandardScaler().fit(X_train_f)
    X_train_s, X_test_s = xs.transform(X_train_f), xs.transform(X_test_f)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s)[:, 1]
    return preds, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--baseline-cond", default="baseline")
    parser.add_argument("--rm-cond", default="reward_modulated")
    parser.add_argument("--label", default="campaign")
    args = parser.parse_args()

    df = load_all(args.probe_dir, args.seeds)
    pop = build_population(df, args.baseline_cond, args.rm_cond)
    theta_cols = get_feature_columns_by_scale(pop)["Theta"]
    print(pop.groupby(["group", "seed"]).size().unstack())

    X = pop[theta_cols].to_numpy(dtype="float64")
    y = pop["has_value"].to_numpy()
    location_id = (
        pop["seed"].astype(str) + "_" + pop["group"] + "_"
        + pop["env_id"].astype(str) + "_" + pop["state_id"].astype(str)
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(np.arange(len(pop)), y, groups=location_id))
    overlap = set(location_id.to_numpy()[train_idx]) & set(location_id.to_numpy()[test_idx])
    assert not overlap

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = pop["group"].to_numpy()[test_idx]
    seeds_test = pop["seed"].to_numpy()[test_idx]

    print(f"\ntrain n={len(train_idx)}  test n={len(test_idx)}")
    preds, probs = fit_eval(X_train, y_train, X_test)
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    print(f"accuracy={acc:.3f}  balanced_accuracy={bal_acc:.3f}  AUC={auc:.3f}")
    print(pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]))

    rng = np.random.default_rng(SEED)
    null_bal_acc = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        y_train_perm = rng.permutation(y_train)
        preds_perm, _ = fit_eval(X_train, y_train_perm, X_test)
        null_bal_acc[i] = balanced_accuracy_score(y_test, preds_perm)
    p_value = (np.sum(null_bal_acc >= bal_acc) + 1) / (N_PERMUTATIONS + 1)
    print(f"permutation null mean={null_bal_acc.mean():.3f}  p={p_value:.4g}")

    print("\n=== predicted positive rate by group (pooled) ===")
    group_rows = []
    for g in ["rm_real", "rm_control", "bl_real", "bl_control"]:
        mask = groups_test == g
        rate = preds[mask].mean() if mask.sum() else float("nan")
        group_rows.append({"seed": "all", "group": g, "n": int(mask.sum()), "rate": rate})
        print(f"  {g:12s} n={mask.sum():4d}  rate={rate:.3f}")

    print("\n=== per-seed ===")
    for seed in args.seeds:
        for g in ["rm_real", "rm_control", "bl_real", "bl_control"]:
            mask = (groups_test == g) & (seeds_test == seed)
            n = int(mask.sum())
            if n:
                rate = preds[mask].mean()
                group_rows.append({"seed": seed, "group": g, "n": n, "rate": rate})
                print(f"  seed={seed} {g:12s} n={n:4d}  rate={rate:.3f}")

    out_dir = os.path.dirname(__file__)
    pd.DataFrame(group_rows).to_csv(os.path.join(out_dir, f"results_probe_blreal_{args.label}.csv"), index=False)
    pd.DataFrame([{"accuracy": acc, "balanced_accuracy": bal_acc, "auc": auc, "p_value": p_value}]).to_csv(
        os.path.join(out_dir, f"results_probe_blreal_overall_{args.label}.csv"), index=False
    )
    print(f"\nSaved results for {args.label}")


if __name__ == "__main__":
    main()
