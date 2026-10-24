"""Shared probing logic for probe_value_off_allenvs.py and
probe_value_on_allenvs.py: load a checkpoint's trained weights exactly as
tem_probe_eval_allenvs.py does, run a fresh frozen-weight all-envs walk
(same protocol for both scripts - same episode count, warmup, and periodic
control-landmark rotation), and return a per-state DataFrame with
is_landmark/is_control_landmark/is_reward/x/y/p_<freq>_<idx> columns.

The only difference between the two callers is whether `agent.use_reward`
is forced to False after construction (value OFF: trained weights kept,
value injection into x_c suppressed) or left at the checkpoint's own
trained setting (value ON: True for reward_modulated/reward_modulated_lc,
already False for baseline/baseline_lc - i.e. "as trained"). Controlling
every other detail of the walk keeps that the only thing that differs
between the two datasets.
"""
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tem_probe_eval_allenvs import _load_pickle_safe, _load_model_from_save, col_names
from _tem_eval import run_multienv_probe

CAMPAIGNS = {
    "baseline": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj",
        "baseline",
    ),
    "rm": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj",
        "reward_modulated",
    ),
    "baseline_lc": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_lc",
        "baseline_lc",
    ),
    "rm_lc": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_lc",
        "reward_modulated_lc",
    ),
}
SEEDS = [42, 7, 1, 2, 3]
N_EPISODES = 2000
CHUNK_EPISODES = 10
WARMUP_STEPS = 500
# Matches the ctrlshuffle25_rewardobj campaigns' own training-time interval -
# applied to ALL 4 conditions uniformly (including baseline/baseline_lc,
# which never carry value but were trained under the same periodic control-
# landmark reshuffling), so this fresh probe's control-landmark dynamics
# match what every checkpoint actually saw during training.
CONTROL_SHUFFLE_INTERVAL = 25


def probe_one(label, campaign_root, condition, seed, force_value_off):
    probe_seed = seed
    save_path = os.path.join(campaign_root, f"seed_{seed}", condition)
    agent_path = os.path.join(save_path, "agent")
    tag = "value OFF" if force_value_off else "value ON (as trained)"
    print(f"\n{'=' * 60}\nProbing ({tag}): {label}  ({save_path})\n{'=' * 60}", flush=True)

    training_dict = _load_pickle_safe(os.path.join(save_path, "params.dict"))
    agent_params = dict(training_dict["agent_params"])
    state_dict = _load_pickle_safe(agent_path)
    _load_model_from_save(save_path)

    env = training_dict["env_class"](**training_dict["env_params"])
    agent = training_dict["agent_class"](**agent_params)
    agent.tem.load_state_dict(state_dict)
    agent.tem.eval()

    if force_value_off:
        # Trained weights stay as loaded; value injection into model_input
        # is switched off for this probe regardless of what the checkpoint
        # was trained with.
        agent.use_reward = False
    # else: leave agent.use_reward exactly as agent_params set it (True for
    # reward_modulated/reward_modulated_lc, already False for the other two).

    random.seed(probe_seed)
    np.random.seed(probe_seed)
    obs, state = env.reset(random_state=False, custom_state=[0, 0])

    rates_to_save, env_meta = run_multienv_probe(
        agent, env, obs,
        n_episodes=N_EPISODES,
        chunk_episodes=CHUNK_EPISODES,
        warmup_steps=WARMUP_STEPS,
        progress_every=20,
        progress_prefix=f"[{label}] ",
        save_dir=None,
        condition=condition,
        control_shuffle_interval=CONTROL_SHUFFLE_INTERVAL,
    )

    n_envs = agent.batch_size
    n_p_list = agent.pars["n_p"]
    room_widths = env_meta["room_widths"]
    room_depths = env_meta["room_depths"]
    reward_location = np.array(env_meta["reward_location"])

    rows = []
    for j in range(n_envs):
        room_w = int(room_widths[j])
        room_d = int(room_depths[j])
        n_states_j = room_w * room_d
        x_array = np.linspace(-room_w / 2 + 0.5, room_w / 2 - 0.5, room_w)
        y_array = np.linspace(-room_d / 2 + 0.5, room_d / 2 - 0.5, room_d)

        def state_xy(sid, x_array=x_array, y_array=y_array, room_w=room_w):
            row, col = sid // room_w, sid % room_w
            return x_array[col], y_array[row]

        all_xy = np.array([state_xy(s) for s in range(n_states_j)])
        reward_state_id = int(np.argmin(np.linalg.norm(all_xy - reward_location, axis=1)))

        env_j = env.environments[j]
        object_layout = env_j.objects
        object_ids_per_state = np.argmax(object_layout, axis=1)
        is_landmark = (object_ids_per_state < env_j.n_landmarks).astype(int)
        n_special = env_j.n_landmarks + env_j.n_control_landmarks
        is_control_landmark = (
            (object_ids_per_state >= env_j.n_landmarks) & (object_ids_per_state < n_special)
        ).astype(int)

        p_all = rates_to_save[f"env{j}"]
        xs, ys = zip(*[state_xy(s) for s in range(n_states_j)])

        meta_df = pd.DataFrame({
            "condition": label,
            "env_id": j,
            "state_id": np.arange(n_states_j),
            "x": xs,
            "y": ys,
            "is_landmark": is_landmark,
            "is_control_landmark": is_control_landmark,
            "is_reward": (np.arange(n_states_j) == reward_state_id).astype(int),
            "value": np.nan,
        })
        feat_df = pd.DataFrame(p_all, columns=col_names(n_p_list))
        rows.append(pd.concat([meta_df, feat_df], axis=1))

    df = pd.concat(rows, ignore_index=True)
    print(f"  [{label}] Done. {df.shape}", flush=True)
    return df


def run_all(out_dir, suffix, force_value_off, overwrite=False):
    os.makedirs(out_dir, exist_ok=True)
    for seed in SEEDS:
        for label, (root, condition) in CAMPAIGNS.items():
            out_path = os.path.join(out_dir, f"{label}_seed{seed}_{suffix}.csv")
            if os.path.exists(out_path) and not overwrite:
                print(f"skip (exists): {out_path}")
                continue
            df = probe_one(label, root, condition, seed, force_value_off)
            df.to_csv(out_path, index=False)
            print(f"Saved {out_path}")
        print(f"### seed {seed} done ###", flush=True)
