"""Value decoding for the loop-shift experiment: the scripted loop only
ever visits 14 of a room's ~80-144 states, so control landmarks (placed
randomly room-wide, not path-constrained) are sparse-to-absent on the loop
itself - not enough to power the has_value classifier's negative-control
group. Instead of decoding from the loop walk's own sparse visits, this
does a dedicated whole-room snapshot pass (reusing run_multienv_probe,
unrestricted - same machinery as probe_value_allenvs_common.py) for the
PRE-shift and POST-shift landmark/reward layouts separately: two full
environment constructions per condition/seed, one with landmark_positions
= the original approach-path states, one with landmark_positions = the
same states shifted back 1 loop-state (matching probe_loop_shift.py's
shift exactly) - then the ordinary has_value_full_training_test.py recipe
(Theta features, baseline mixed in as negative, group-safe split,
permutation test) on each, so "does value decodability survive the
landmark/reward relocation" can be read directly off the pre vs post
numbers for each {rotation, LC} pair.

Saves one allenvs_place_cells-style CSV per (condition, seed, half) under
Thesis/quantitative/loop/place_fields/, plus the final has_value summary
tables under Thesis/quantitative/loop/.
"""
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

AGENT_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AGENT_EXAMPLES_DIR)
from tem_probe_eval_allenvs import _load_pickle_safe, _load_model_from_save, col_names  # noqa: E402
from _tem_eval import run_multienv_probe  # noqa: E402
from probe_loop_shift import (  # noqa: E402
    SQUARE_LOOP, LOOP_START, _simulate_path, _nearest_state_xy,
)

REWARD_PRED_SCRIPTS = r"C:\Users\hansl\Desktop\Projects\Masters_research\reward_prediction\scripts"
sys.path.insert(0, REWARD_PRED_SCRIPTS)
from common_decode import SEED, _drop_low_variance, get_feature_columns_by_scale  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

THESIS = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis"
LOOP_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop"
PLACE_FIELDS_DIR = os.path.join(LOOP_DIR, "place_fields")
SEEDS = [42, 7, 1, 2, 3]
N_EPISODES = 500
CHUNK_EPISODES = 10
WARMUP_STEPS = 200
N_PERMUTATIONS = 300

CAMPAIGNS = {
    "baseline": (os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj"), "baseline"),
    "rm": (os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj"), "reward_modulated"),
    "baseline_lc": (os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_lc"), "baseline_lc"),
    "rm_lc": (os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_lc"), "reward_modulated_lc"),
    "baseline_noshuffle": (os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle"), "baseline"),
    "rm_noshuffle": (os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle"), "reward_modulated"),
    "baseline_lc_noshuffle": (os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle_lc"), "baseline_lc"),
    "rm_lc_noshuffle": (os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle_lc"), "reward_modulated_lc"),
}
PAIRS = {
    "rotation ON, no LC": ("baseline", "rm"),
    "rotation ON, LC": ("baseline_lc", "rm_lc"),
    "rotation OFF, no LC": ("baseline_noshuffle", "rm_noshuffle"),
    "rotation OFF, LC": ("baseline_lc_noshuffle", "rm_lc_noshuffle"),
}


def loop_landmark_positions(save_path, shifted):
    training_dict = pd.read_pickle(os.path.join(save_path, "params.dict"))
    agent_params = training_dict["agent_params"]
    discrete_env_params = training_dict["env_params"]["arg_env_params"]
    room_w, room_d = agent_params["room_widths"][0], agent_params["room_depths"][0]
    reward_location = discrete_env_params["reward_location"]

    full_path = [LOOP_START] + _simulate_path(LOOP_START, SQUARE_LOOP, step_size=1)
    unique_path_states = full_path[:-1]
    n_loop_states = len(unique_path_states)
    reward_xy = _nearest_state_xy(reward_location, room_w, room_d)
    reward_idx = next(i for i, p in enumerate(unique_path_states) if np.allclose(p, reward_xy))
    approach_states = unique_path_states[: reward_idx + 1]
    n_landmarks = discrete_env_params.get("n_landmarks", 10)
    positions = approach_states[:n_landmarks]
    if shifted:
        positions = [unique_path_states[(i - 1) % n_loop_states] for i in range(len(positions))]
    return positions


def probe_one(label, save_path, seed, half):
    shifted = half == "post"
    print(f"\n[{label} seed={seed} {half}] probing...", flush=True)
    training_dict = _load_pickle_safe(os.path.join(save_path, "params.dict"))
    agent_params = dict(training_dict["agent_params"])
    discrete_env_params = dict(training_dict["env_params"]["arg_env_params"])
    discrete_env_params["landmark_positions"] = loop_landmark_positions(save_path, shifted)
    env_params = dict(training_dict["env_params"])
    env_params["arg_env_params"] = discrete_env_params
    state_dict = _load_pickle_safe(os.path.join(save_path, "agent"))
    _load_model_from_save(save_path)

    env = training_dict["env_class"](**env_params)
    agent = training_dict["agent_class"](**agent_params)
    agent.tem.load_state_dict(state_dict)
    agent.tem.eval()

    random.seed(seed)
    np.random.seed(seed)
    obs, state = env.reset(random_state=False, custom_state=[0, 0])

    rates_to_save, env_meta = run_multienv_probe(
        agent, env, obs, n_episodes=N_EPISODES, chunk_episodes=CHUNK_EPISODES,
        warmup_steps=WARMUP_STEPS, progress_every=50, progress_prefix=f"[{label} {half}] ",
        save_dir=None, condition=label,
    )

    n_envs = agent.batch_size
    n_p_list = agent.pars["n_p"]
    room_widths, room_depths = env_meta["room_widths"], env_meta["room_depths"]
    reward_location = np.array(env_meta["reward_location"])

    rows = []
    for j in range(n_envs):
        room_w, room_d = int(room_widths[j]), int(room_depths[j])
        n_states_j = room_w * room_d
        x_array = np.linspace(-room_w / 2 + 0.5, room_w / 2 - 0.5, room_w)
        y_array = np.linspace(-room_d / 2 + 0.5, room_d / 2 - 0.5, room_d)

        def state_xy(sid, x_array=x_array, y_array=y_array, room_w=room_w):
            row, col = sid // room_w, sid % room_w
            return x_array[col], y_array[row]

        all_xy = np.array([state_xy(s) for s in range(n_states_j)])
        reward_state_id = int(np.argmin(np.linalg.norm(all_xy - reward_location, axis=1)))

        env_j = env.environments[j]
        object_ids_per_state = np.argmax(env_j.objects, axis=1)
        is_landmark = (object_ids_per_state < env_j.n_landmarks).astype(int)
        n_special = env_j.n_landmarks + env_j.n_control_landmarks
        is_control_landmark = ((object_ids_per_state >= env_j.n_landmarks) & (object_ids_per_state < n_special)).astype(int)

        p_all = rates_to_save[f"env{j}"]
        xs, ys = zip(*[state_xy(s) for s in range(n_states_j)])
        meta_df = pd.DataFrame({
            "condition": label, "env_id": j, "state_id": np.arange(n_states_j), "x": xs, "y": ys,
            "is_landmark": is_landmark, "is_control_landmark": is_control_landmark,
            "is_reward": (np.arange(n_states_j) == reward_state_id).astype(int),
        })
        feat_df = pd.DataFrame(p_all, columns=col_names(n_p_list))
        rows.append(pd.concat([meta_df, feat_df], axis=1))

    df = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(PLACE_FIELDS_DIR, f"{label}_seed{seed}_{half}.csv")
    df.to_csv(out_path, index=False)
    print(f"[{label} seed={seed} {half}] saved {out_path} {df.shape}")


def fit_eval(X_train, y_train, X_test, y_test):
    X_train_f, X_test_f, keep = _drop_low_variance(X_train, X_test, verbose=False)
    xs = StandardScaler().fit(X_train_f)
    X_train_s, X_test_s = xs.transform(X_train_f), xs.transform(X_test_f)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_s, y_train)
    return clf.predict(X_test_s), clf.predict_proba(X_test_s)[:, 1]


def has_value_test(test_label, baseline_label, rm_label, half):
    frames = []
    for seed in SEEDS:
        for label in (baseline_label, rm_label):
            df = pd.read_csv(os.path.join(PLACE_FIELDS_DIR, f"{label}_seed{seed}_{half}.csv"))
            df["seed"] = seed
            frames.append(df)
    pop_all = pd.concat(frames, ignore_index=True)
    theta_cols = get_feature_columns_by_scale(pop_all)["Theta"]

    rm = pop_all[pop_all["condition"] == rm_label]
    bl = pop_all[pop_all["condition"] == baseline_label]
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
                   + pop["env_id"].astype(str) + "_" + pop["state_id"].astype(str))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(np.arange(len(pop)), y, groups=location_id))
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

    print(f"[{test_label} / {half}] acc={acc:.3f} bal_acc={bal_acc:.3f} auc={auc:.3f} p={p_value:.4g}  "
          f"rm_real={group_rates['rm_real']:.3f} rm_control={group_rates['rm_control']:.3f} "
          f"bl_real={group_rates['bl_real']:.3f} bl_control={group_rates['bl_control']:.3f}")

    return {
        "test": test_label, "half": half, "accuracy": acc, "balanced_accuracy": bal_acc, "auc": auc,
        "permutation_p_value": p_value, **{f"rate_{g}": r for g, r in group_rates.items()},
    }


def main():
    os.makedirs(PLACE_FIELDS_DIR, exist_ok=True)
    for seed in SEEDS:
        for label, (root, condition) in CAMPAIGNS.items():
            save_path = os.path.join(root, f"seed_{seed}", condition)
            for half in ("pre", "post"):
                out_path = os.path.join(PLACE_FIELDS_DIR, f"{label}_seed{seed}_{half}.csv")
                if os.path.exists(out_path):
                    print(f"skip (exists): {out_path}")
                    continue
                probe_one(label, save_path, seed, half)
        print(f"### seed {seed} done ###", flush=True)

    results = []
    for test_label, (baseline_label, rm_label) in PAIRS.items():
        for half in ("pre", "post"):
            results.append(has_value_test(test_label, baseline_label, rm_label, half))
    out_df = pd.DataFrame(results)
    out_df.to_csv(os.path.join(LOOP_DIR, "value_decoding_pre_post.csv"), index=False)
    print(f"\nSaved {os.path.join(LOOP_DIR, 'value_decoding_pre_post.csv')}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
