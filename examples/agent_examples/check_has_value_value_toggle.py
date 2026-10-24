"""Apply the project's actual established has_value methodology (not a
quick ad-hoc Ridge regression) to the new value-ON/value-OFF probe CSVs -
reusing reward_prediction/scripts/value_decode_probe.py's exact recipe,
itself adapted from has_value_full_training_test.py: train the classifier
WITH baseline's real landmarks included as explicit NEGATIVES (alongside
reward_modulated's own control landmarks and baseline's own controls), so
shape/identity alone can no longer separate the classes - only an actual
value-specific signal can. Theta-scale features only (matches that
methodology exactly, not the full 5-scale/440-dim feature set this
session's other scripts used), LogisticRegression + StandardScaler +
drop-low-variance, GroupShuffleSplit held out by physical (seed, group,
env_id, state_id) location, 300-permutation null test on balanced accuracy.

Difference from the original scripts: this runs on probe_value_off_allenvs.py/
probe_value_on_allenvs.py's fresh CSVs (which don't populate a per-row
`value` column - that was an oversight in this session's refactor, not worth
re-running the ~8h probe just to fix), so build_population() here drops the
`value.notna()` filter and uses `is_landmark == 1` alone to define rm_real/
bl_real. With 2000 episodes of exposure per environment, this is not
expected to change anything - TD(0) learns online from the very first visit,
so by the end of the walk essentially every landmark id has a learned value
in the original pipeline too; the notna() filter there was a validity
sanity-check, not a meaningful exclusion criterion.

Runs 4 separate has_value tests (each its own population/classifier/split):
non-LC value-OFF, non-LC value-ON, LC value-OFF, LC value-ON - so the
has_value effect of the value toggle can be read directly off the same
accuracy/balanced_accuracy/AUC/group-rate numbers already used throughout
this project.
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

PROBE_ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 7, 1, 2, 3]
N_PERMUTATIONS = 300

CAMPAIGNS = {
    "non-LC": {"baseline": "baseline", "rm": "rm"},
    "LC": {"baseline": "baseline_lc", "rm": "rm_lc"},
}


def load_all(probe_dir, suffix, baseline_label, rm_label):
    frames = []
    for seed in SEEDS:
        for label in [baseline_label, rm_label]:
            path = os.path.join(probe_dir, f"{label}_seed{seed}_{suffix}.csv")
            df = pd.read_csv(path)
            df["seed"] = seed
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_population(df, baseline_label, rm_label):
    rm = df[df["condition"] == rm_label]
    bl = df[df["condition"] == baseline_label]

    rm_real = rm[rm["is_landmark"] == 1].copy()
    rm_control = rm[rm["is_control_landmark"] == 1].copy()
    bl_real = bl[bl["is_landmark"] == 1].copy()
    bl_control = bl[bl["is_control_landmark"] == 1].copy()

    rm_real["group"] = "rm_real"
    rm_control["group"] = "rm_control"
    bl_real["group"] = "bl_real"
    bl_control["group"] = "bl_control"

    pop = pd.concat([rm_real, rm_control, bl_real, bl_control], ignore_index=True)
    pop["has_value"] = (pop["group"] == "rm_real").astype(int)
    return pop


def fit_eval(X_train, y_train, X_test, y_test):
    X_train_f, X_test_f, keep = _drop_low_variance(X_train, X_test, verbose=False)
    xs = StandardScaler().fit(X_train_f)
    X_train_s, X_test_s = xs.transform(X_train_f), xs.transform(X_test_f)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s)[:, 1]
    return preds, probs


def run_test(label, probe_dir, suffix, baseline_label, rm_label):
    print(f"\n{'=' * 70}\nhas_value test: {label}\n{'=' * 70}")
    df = load_all(probe_dir, suffix, baseline_label, rm_label)
    pop = build_population(df, baseline_label, rm_label)
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
    assert not overlap, f"{len(overlap)} locations leaked across the train/test split"

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = pop["group"].to_numpy()[test_idx]

    print(f"train n={len(train_idx)}  test (held out) n={len(test_idx)}")

    preds, probs = fit_eval(X_train, y_train, X_test, y_test)
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    print(f"held-out test: accuracy={acc:.3f}  balanced_accuracy={bal_acc:.3f}  AUC={auc:.3f}")
    print(pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]))

    print(f"running {N_PERMUTATIONS}-permutation null test on held-out balanced_accuracy...")
    rng = np.random.default_rng(SEED)
    null_bal_acc = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        y_train_perm = rng.permutation(y_train)
        preds_perm, _ = fit_eval(X_train, y_train_perm, X_test, y_test)
        null_bal_acc[i] = balanced_accuracy_score(y_test, preds_perm)
    p_value = (np.sum(null_bal_acc >= bal_acc) + 1) / (N_PERMUTATIONS + 1)
    print(f"null mean={null_bal_acc.mean():.3f} std={null_bal_acc.std():.3f}  p={p_value:.4g}")

    group_rows = []
    print("predicted has_value=1 rate by group (held-out test):")
    for g in ["rm_real", "rm_control", "bl_real", "bl_control"]:
        mask = groups_test == g
        rate = preds[mask].mean() if mask.sum() else float("nan")
        group_rows.append({"test": label, "group": g, "n": int(mask.sum()), "predicted_positive_rate": rate})
        print(f"  {g:12s} n={mask.sum():4d}  predicted positive rate={rate:.3f}")

    return {
        "test": label, "accuracy": acc, "balanced_accuracy": bal_acc, "auc": auc,
        "permutation_null_mean": null_bal_acc.mean(), "permutation_null_std": null_bal_acc.std(),
        "permutation_p_value": p_value, "n_train": len(train_idx), "n_test": len(test_idx),
    }, group_rows


def main():
    overall_rows, group_rows_all = [], []
    for campaign, labels in CAMPAIGNS.items():
        for probe_dir, suffix, value_tag in [
            (os.path.join(PROBE_ROOT, "value_off_probe"), "valueoff", "value OFF"),
            (os.path.join(PROBE_ROOT, "value_on_probe"), "valueon", "value ON"),
        ]:
            label = f"{campaign} {value_tag}"
            overall, group_rows = run_test(label, probe_dir, suffix, labels["baseline"], labels["rm"])
            overall_rows.append(overall)
            group_rows_all.extend(group_rows)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    overall_df = pd.DataFrame(overall_rows)
    print(overall_df.to_string(index=False))

    group_df = pd.DataFrame(group_rows_all)
    print("\nper-group predicted-positive rates:")
    print(group_df.pivot(index="group", columns="test", values="predicted_positive_rate").reindex(
        ["rm_real", "rm_control", "bl_real", "bl_control"]
    ).to_string())

    overall_df.to_csv(os.path.join(PROBE_ROOT, "has_value_value_toggle_overall.csv"), index=False)
    group_df.to_csv(os.path.join(PROBE_ROOT, "has_value_value_toggle_groups.csv"), index=False)
    print(f"\nSaved has_value_value_toggle_overall.csv and has_value_value_toggle_groups.csv to {PROBE_ROOT}")


if __name__ == "__main__":
    main()
