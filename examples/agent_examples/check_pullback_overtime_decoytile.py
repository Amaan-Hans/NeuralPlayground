"""Same as check_pullback_overtime.py, pointed at the decoy-tiled campaign's
checkpoints (Thesis2 / *_decoytile) and its already-computed *_place_cells.csv
settled reference (from probe_loop_ring_all_decoytile.py) instead of the
original campaign's. Saves pullback_overtime_episodes.csv and
pullback_overtime_baseline.csv to
Thesis2/quantitative/loop/overrepresentation_experiment/.
"""
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

AGENT_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AGENT_EXAMPLES_DIR)
from probe_loop_ring import (  # noqa: E402
    build_ring_env_agent, place_ring_objects, pick_control_states, apply_shift,
    snapshot_ring_categories,
    BATCH_SIZE, RING_LOOP, LOOP_START, TRAJECTORY_SEED,
)
from probe_loop_ring_all_decoytile import CAMPAIGNS  # noqa: E402

THESIS2 = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis2"
LOOP_DIR = os.path.join(THESIS2, "quantitative", "loop")
OUT_DIR = os.path.join(LOOP_DIR, "overrepresentation_experiment")
SEEDS = [42, 7, 1, 2, 3]
EARLY_WINDOW = 10


def run_transient_short(save_path, label, seed):
    cfg = build_ring_env_agent(save_path)
    env, agent = cfg["env"], cfg["agent"]
    n_envs = BATCH_SIZE
    n_rollout = agent.pars["n_rollout"]
    n_ring_states = cfg["n_ring_states"]
    n_landmarks, n_special = cfg["n_landmarks"], cfg["n_special"]
    reward_object_id = cfg["reward_object_id"]
    n_p_list = agent.pars["n_p"]
    total_p = sum(n_p_list)

    random.seed(TRAJECTORY_SEED)
    np.random.seed(TRAJECTORY_SEED)
    obs, state = env.reset(random_state=False, custom_state=list(LOOP_START))
    control_states = pick_control_states(cfg["non_value_states"], random)
    place_ring_objects(env, cfg, control_states)
    obs = [env.environments[j].make_object_observation(np.array(LOOP_START)) for j in range(BATCH_SIZE)]
    agent.n_walk = 0
    agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

    loop_idx = 0
    prev_iter = None
    ring_categories = {"pre": snapshot_ring_categories(env, cfg, n_landmarks, n_special, reward_object_id), "post": None}
    episode_rows = []

    with torch.no_grad():
        for local_ep in range(2 * EARLY_WINDOW):
            era = "early_pre" if local_ep < EARLY_WINDOW else "early_post"
            half = "pre" if local_ep < EARLY_WINDOW else "post"

            loop_idx_log = []
            while agent.n_walk < n_rollout:
                loop_idx_log.append(loop_idx % n_ring_states)
                actions = agent.batch_act(obs)
                loop_action = RING_LOOP[loop_idx % len(RING_LOOP)]
                env0_state_pre = obs[0][0]
                for i in range(n_envs):
                    actions[i] = loop_action
                    agent.prev_actions[i] = loop_action
                obs, state, reward = env.step(actions, normalize_step=True)
                if obs[0][0] != env0_state_pre:
                    loop_idx += 1
            agent.n_walk = 0

            real_indices = [i for i, step in enumerate(agent.obs_history) if step[0][0] != -1]
            history = [agent.obs_history[i] for i in real_indices]
            actions_raw = [agent.walk_actions[i] for i in real_indices]
            held_raw = [agent.held_landmark_history[i] for i in real_indices]
            pos_at_step = [(loop_idx_log[i] - 1) % n_ring_states for i in real_indices]
            chunk_len = len(history)
            if chunk_len == 0:
                agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []
                continue

            locations = [[{"id": step[j][0], "shiny": None} for j in range(n_envs)] for step in history]
            observations = [[step[j][1] for j in range(n_envs)] for step in history]
            action_values = agent.step_to_actions(actions_raw)
            obs_array = np.reshape(observations, (chunk_len, n_envs, agent.pars["n_x"]))
            action_array = np.reshape(action_values, (chunk_len, n_envs))

            v_seq = None
            if agent.use_reward and agent.td is not None:
                v_seq = agent._value_for_history(held_raw)

            model_input = []
            for i in range(chunk_len):
                step = [locations[i], torch.from_numpy(obs_array[i]).type(torch.float32).to(agent.device),
                        action_array[i].tolist()]
                if v_seq is not None:
                    step.append(None)
                    step.append(v_seq[i])
                model_input.append(step)

            forward = agent.tem(model_input, prev_iter)

            ep_sum = np.zeros((n_envs, n_ring_states, total_p))
            ep_count = np.zeros((n_envs, n_ring_states), dtype=np.int64)
            for local_i, step in enumerate(forward):
                pos = pos_at_step[local_i]
                p_vec_all = np.concatenate([step.p_inf[f].detach().cpu().numpy() for f in range(len(n_p_list))], axis=-1)
                for j in range(n_envs):
                    ep_sum[j, pos] += p_vec_all[j]
                    ep_count[j, pos] += 1

            prev_iter = [forward[-1]]
            agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

            cats = ring_categories[half]
            pos_total = np.array([
                (ep_sum[:, pos].sum(axis=-1) / np.maximum(ep_count[:, pos], 1)).mean()
                if ep_count[:, pos].sum() > 0 else np.nan
                for pos in range(n_ring_states)
            ])
            denom = np.nansum(pos_total)
            shares = pos_total / denom if denom > 0 else np.full(n_ring_states, np.nan)
            landmark_positions = sorted(p for p in range(n_ring_states) if cats[p] == "landmark")
            reward_position = next(p for p in range(n_ring_states) if cats[p] == "reward")
            row = {"condition": label, "seed": seed, "episode": local_ep % EARLY_WINDOW, "era": era,
                   "reward_share": shares[reward_position]}
            for i, p in enumerate(landmark_positions):
                row[f"landmark{i}_share"] = shares[p]
            for k in range(n_ring_states):
                row[f"k{k}_activity"] = pos_total[(reward_position - k) % n_ring_states]
            episode_rows.append(row)

            if local_ep + 1 == EARLY_WINDOW:
                apply_shift(env, agent, cfg)
                ring_categories["post"] = snapshot_ring_categories(env, cfg, n_landmarks, n_special, reward_object_id)

    print(f"[{label} seed {seed}] done", flush=True)
    return pd.DataFrame(episode_rows)


def compute_baseline_shares(cond, seed, half):
    path = os.path.join(LOOP_DIR, f"{cond}_seed{seed}_place_cells.csv")
    df = pd.read_csv(path)
    dfh = df[df["half"] == half]
    p_cols = [c for c in dfh.columns if c.startswith("p_")]
    grouped = dfh.groupby("ring_pos")[p_cols].mean()
    pos_total = grouped.sum(axis=1)
    shares = pos_total / pos_total.sum()
    cats = dfh.groupby("ring_pos").first()[["is_landmark", "is_control_landmark", "is_reward"]]
    landmark_positions = sorted(cats.index[cats["is_landmark"] == 1].tolist())
    reward_position = cats.index[cats["is_reward"] == 1][0]
    n_ring_states = dfh["ring_pos"].nunique()
    row = {"condition": cond, "seed": seed, "half": half, "reward_share": shares.get(reward_position, np.nan)}
    for i, p in enumerate(landmark_positions):
        row[f"landmark{i}_share"] = shares.get(p, np.nan)
    for k in range(n_ring_states):
        pos = (reward_position - k) % n_ring_states
        row[f"k{k}_activity"] = pos_total.get(pos, np.nan)
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_episodes, all_baseline = [], []
    for cond, (root, condition) in CAMPAIGNS.items():
        for seed in SEEDS:
            save_path = os.path.join(root, f"seed_{seed}", condition)
            print(f"=== {cond} seed {seed} ===", flush=True)
            ep_df = run_transient_short(save_path, cond, seed)
            all_episodes.append(ep_df)
            for half in ("pre", "post"):
                all_baseline.append(compute_baseline_shares(cond, seed, half))

    pd.concat(all_episodes, ignore_index=True).to_csv(os.path.join(OUT_DIR, "pullback_overtime_episodes.csv"), index=False)
    pd.DataFrame(all_baseline).to_csv(os.path.join(OUT_DIR, "pullback_overtime_baseline.csv"), index=False)
    print("Saved pullback_overtime_episodes.csv and pullback_overtime_baseline.csv")


if __name__ == "__main__":
    main()
