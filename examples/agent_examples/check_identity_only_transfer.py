"""Reverse of check_classifier_transfer.py: train a classifier that has NEVER
seen a value signal at all, then see how it treats actually value-carrying
landmarks.

The no-shuffle baseline campaign has use_reward=False for every seed - no
object anywhere in it ever carries value. So a classifier trained there to
separate is_landmark==1 (bl_real) from is_control_landmark==1 (bl_control)
can only be learning object IDENTITY (landmark-type vs control-type sensory
code), since value was never present as a possible explanation.

Apply that frozen identity-only classifier to the shuffle-enabled campaign's
full population (rm_real/rm_control/bl_real/bl_control). If rm_real (which
DOES carry real injected value) gets flagged at a similar rate to bl_real
(landmark-type, no value) under this identity-only classifier, that is
strong evidence the original has_value classifier's "detection" of rm_real
was mostly just identity, not anything specific to the injected value
signal. If rm_real is flagged much more than bl_real even here, injected
value is doing something identity alone can't explain.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REWARD_PRED_SCRIPTS = r"C:\Users\hansl\Desktop\Projects\Masters_research\reward_prediction\scripts"
sys.path.insert(0, REWARD_PRED_SCRIPTS)
from common_decode import _drop_low_variance, get_feature_columns_by_scale  # noqa: E402

TRAIN_ROOT = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle"
TRAIN_SEEDS = [42, 7, 2, 3]  # seed 1 baseline still training
TEST_ROOT = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj"
TEST_SEEDS = [42, 7, 1, 2, 3]


def load_identity_only_population():
    """No-shuffle baseline: is_landmark vs is_control_landmark, zero value anywhere."""
    frames = []
    for seed in TRAIN_SEEDS:
        path = os.path.join(TRAIN_ROOT, f"seed_{seed}", "baseline", "allenvs_place_cells.csv")
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    landmark = df[df["is_landmark"] == 1].copy()
    control = df[df["is_control_landmark"] == 1].copy()
    landmark["group"], control["group"] = "bl_real", "bl_control"
    landmark["is_landmark_identity"] = 1
    control["is_landmark_identity"] = 0
    return pd.concat([landmark, control], ignore_index=True)


def load_shuffle_enabled_population():
    frames = []
    for seed in TEST_SEEDS:
        for condition in ["baseline", "reward_modulated"]:
            path = os.path.join(TEST_ROOT, f"seed_{seed}", condition, "allenvs_place_cells.csv")
            df = pd.read_csv(path)
            df["seed"] = seed
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    rm = df[df["condition"] == "reward_modulated"]
    bl = df[df["condition"] == "baseline"]
    rm_real = rm[(rm["is_landmark"] == 1) & rm["value"].notna()].copy()
    rm_control = rm[rm["is_control_landmark"] == 1].copy()
    bl_real = bl[bl["is_landmark"] == 1].copy()
    bl_control = bl[bl["is_control_landmark"] == 1].copy()
    rm_real["group"], rm_control["group"] = "rm_real", "rm_control"
    bl_real["group"], bl_control["group"] = "bl_real", "bl_control"
    return pd.concat([rm_real, rm_control, bl_real, bl_control], ignore_index=True)


def main():
    train_pop = load_identity_only_population()
    test_pop = load_shuffle_enabled_population()

    theta_cols = get_feature_columns_by_scale(train_pop)["Theta"]
    missing = [c for c in theta_cols if c not in test_pop.columns]
    if missing:
        print(f"WARNING: {len(missing)} theta columns missing in test population.")
        theta_cols = [c for c in theta_cols if c in test_pop.columns]

    X_train = train_pop[theta_cols].to_numpy(dtype="float64")
    y_train = train_pop["is_landmark_identity"].to_numpy()
    X_test = test_pop[theta_cols].to_numpy(dtype="float64")

    print(f"Fitting an IDENTITY-ONLY classifier on the no-shuffle baseline "
          f"(n={len(train_pop)}) - this population has zero value anywhere, so it can only "
          f"be learning landmark-type vs control-type object identity...")
    X_train_f, X_test_f, keep = _drop_low_variance(X_train, X_test, verbose=False)
    scaler = StandardScaler().fit(X_train_f)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X_train_f), y_train)

    home_probs = clf.predict_proba(scaler.transform(X_train_f))[:, 1]
    home_preds = home_probs > 0.5
    print("\n=== in-sample rate on its own home population (sanity check) ===")
    for g in ["bl_real", "bl_control"]:
        mask = (train_pop["group"] == g).to_numpy()
        print(f"  {g:12s} n={mask.sum():4d}  predicted 'landmark-identity' rate={home_preds[mask].mean():.3f}")

    test_probs = clf.predict_proba(scaler.transform(X_test_f))[:, 1]
    test_preds = test_probs > 0.5
    print(f"\n=== transferred to the shuffle-enabled campaign (n={len(test_pop)}, never seen, "
          f"includes REAL injected value in rm_real) ===")
    for g in ["rm_real", "rm_control", "bl_real", "bl_control"]:
        mask = (test_pop["group"] == g).to_numpy()
        rate = test_preds[mask].mean() if mask.sum() else float("nan")
        prob_mean = test_probs[mask].mean() if mask.sum() else float("nan")
        print(f"  {g:12s} n={mask.sum():4d}  predicted 'landmark-identity' rate={rate:.3f}  mean score={prob_mean:.3f}")

    rm_real_rate = test_preds[(test_pop["group"] == "rm_real").to_numpy()].mean()
    bl_real_rate = test_preds[(test_pop["group"] == "bl_real").to_numpy()].mean()
    print(f"\nrm_real vs bl_real under the identity-only classifier: {rm_real_rate:.3f} vs {bl_real_rate:.3f}")
    print("(if these are close, the original has_value classifier's signal on rm_real was "
          "mostly identity, not injected value; if rm_real is clearly higher, injected value "
          "is doing something identity alone doesn't explain)")

    print("\n=== per-seed breakdown, rm_real vs bl_real (shuffle-enabled campaign) ===")
    for seed in TEST_SEEDS:
        for g in ["rm_real", "bl_real"]:
            mask = ((test_pop["group"] == g) & (test_pop["seed"] == seed)).to_numpy()
            if mask.sum():
                print(f"  seed={seed} {g:8s} n={mask.sum():4d}  rate={test_preds[mask].mean():.3f}")

    out = pd.DataFrame([
        {"population": "train_home_noshuffle", "group": g,
         "n": int((train_pop["group"] == g).sum()),
         "predicted_landmark_identity_rate": float(home_preds[(train_pop["group"] == g).to_numpy()].mean())}
        for g in ["bl_real", "bl_control"]
    ] + [
        {"population": "shuffle_enabled_transfer", "group": g,
         "n": int((test_pop["group"] == g).sum()),
         "predicted_landmark_identity_rate": float(test_preds[(test_pop["group"] == g).to_numpy()].mean())}
        for g in ["rm_real", "rm_control", "bl_real", "bl_control"]
    ])
    out_path = os.path.join(os.path.dirname(__file__), "results_identity_only_transfer.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
