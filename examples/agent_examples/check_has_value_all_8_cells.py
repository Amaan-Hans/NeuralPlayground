"""The established has_value methodology (reward_prediction/scripts/
has_value_full_training_test.py's recipe, same as check_has_value_value_toggle.py
used earlier) run on all 4 {rotation, LC} pairs now that every cell's
allenvs_place_cells.csv (the real as-trained final-episode snapshot, not a
fresh re-probe) is downloaded locally - rotation-on/no-LC, rotation-on/LC,
rotation-off/no-LC, rotation-off/LC. 5 seeds pooled per test, same
LogisticRegression + Theta-scale-only features + StandardScaler +
drop-low-variance + GroupShuffleSplit (held out by physical seed/group/
env_id/state_id location) + 300-permutation significance test as every
other has_value result in this project, so the numbers are directly
comparable to the project's historical has_value tables.

Population per test (matches has_value_full_training_test.py's "train WITH
baseline as negatives" design, breaking the shape confound): positive =
that cell's reward_modulated real landmarks; negative = that cell's own
reward_modulated controls + that cell's own baseline real landmarks +
baseline controls.
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

# label -> (campaign_root, baseline_condition_subfolder, rm_condition_subfolder)
CELLS = {
    "rotation ON, no LC": (
        os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj"),
        "baseline", "reward_modulated",
    ),
    "rotation ON, LC": (
        os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_lc"),
        "baseline_lc", "reward_modulated_lc",
    ),
    "rotation OFF, no LC": (
        os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle"),
        "baseline", "reward_modulated",
    ),
    "rotation OFF, LC": (
        os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle_lc"),
        "baseline_lc", "reward_modulated_lc",
    ),
}


def load_all(campaign_root, baseline_cond, rm_cond):
    frames = []
    for seed in SEEDS:
        for cond in (baseline_cond, rm_cond):
            path = os.path.join(campaign_root, f"seed_{seed}", cond, "allenvs_place_cells.csv")
            df = pd.read_csv(path)
            df["seed"] = seed
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_population(df, baseline_cond, rm_cond):
    rm = df[df["condition"] == rm_cond]
    bl = df[df["condition"] == baseline_cond]

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


def run_test(label, campaign_root, baseline_cond, rm_cond):
    print(f"\n{'=' * 70}\nhas_value test: {label}\n{'=' * 70}")
    df = load_all(campaign_root, baseline_cond, rm_cond)
    pop = build_population(df, baseline_cond, rm_cond)
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
    for label, (campaign_root, baseline_cond, rm_cond) in CELLS.items():
        overall, group_rows = run_test(label, campaign_root, baseline_cond, rm_cond)
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

    out_dir = os.path.dirname(os.path.abspath(__file__))
    overall_df.to_csv(os.path.join(out_dir, "has_value_all_8_cells_overall.csv"), index=False)
    group_df.to_csv(os.path.join(out_dir, "has_value_all_8_cells_groups.csv"), index=False)
    print(f"\nSaved has_value_all_8_cells_overall.csv and has_value_all_8_cells_groups.csv to {out_dir}")


if __name__ == "__main__":
    main()
