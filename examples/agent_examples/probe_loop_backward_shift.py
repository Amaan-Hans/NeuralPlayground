"""Frozen-weight probe: force an existing checkpoint's env 0 around a fixed
square loop that passes through the reward-adjacent state (identical
geometry to whittington_2020_loop_run.py's SQUARE_LOOP/LOOP_START), with
VALUE landmarks explicitly placed along the approach to reward (via
`landmark_positions`, same convention) and CONTROL landmarks reshuffled
every 20 raw steps (one n_rollout chunk).

No training occurs anywhere in this script: weights are frozen
(torch.no_grad(), no agent.update()) and the agent's real policy is
irrelevant - every env's action is overwritten with the scripted loop
action each step (same technique whittington_2020_loop_run.py uses for its
phase-2 loop). What's measured is NOT actual movement (that's fixed) but
the model's inferred place-cell code (p_inf) at each point along the loop,
across many repeated laps.

Rationale (why this is testable on a frozen checkpoint at all): the Hebbian
memory M is updated by a hand-coded outer-product rule on every forward
pass, train or eval alike (Model.hebbian()) - it is not a learned weight and
keeps evolving within this probe walk even though the network's weights
never change. If repeated traversal of a fixed, value-marked path leaves any
trace in the inferred representation, it has to show up as a within-walk
drift in p_inf, which is exactly what this script looks for.

Backward-shift metric (the real neuroscience signature this is checking
for - experience-dependent asymmetric expansion of place fields backward,
opposite the direction of travel, e.g. Mehta/O'Keefe 1997): for each lap,
each loop position's p_inf[0] vector is compared (cosine similarity) against
every loop position's p_inf[0] from lap 1 (the reference lap). The
argmax-similarity reference position minus the actual position (circular,
mod loop length) is that position's "shift" for that lap. A developing
NEGATIVE shift over laps = activity for that position increasingly
resembles what used to appear earlier in the path, i.e. shifted backward.
Computed separately for approach-path positions (fixed value landmark) vs
the remaining loop positions (no fixed value landmark there; this loop's
only within-walk source of change) as the comparison.

Usage: python probe_loop_backward_shift.py --save-path <checkpoint dir> --label <name> --out <csv path>
"""
import argparse
import importlib.util
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

AGENT_EXAMPLES_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\NeuralPlayground\examples\agent_examples"
sys.path.insert(0, AGENT_EXAMPLES_DIR)
sys.path.insert(0, os.path.join(AGENT_EXAMPLES_DIR, "..", ".."))
from neuralplayground.agents.whittington_2020 import Whittington2020  # noqa: E402

import __main__ as _main_module  # noqa: E402


def _agent_factory(**kwargs):
    return Whittington2020(**kwargs)


_main_module._agent_factory = _agent_factory

TRAJECTORY_SEED = 42
N_CHUNKS = 30               # 30 * n_rollout(20) = 600 raw steps
CONTROL_RESHUFFLE_EVERY_CHUNK = 1  # reshuffle controls every chunk == every 20 raw steps

# ── Square-loop geometry (identical to whittington_2020_loop_run.py) ──────────
_DOWN, _RIGHT, _UP, _LEFT = [0, -1], [1, 0], [0, 1], [-1, 0]
SQUARE_LOOP = [_DOWN] * 4 + [_RIGHT] * 3 + [_UP] * 4 + [_LEFT] * 3
LOOP_START = (-0.5, -0.5)


def _simulate_path(start, actions, step_size=1):
    """Replicate DiscreteObjectEnvironment.step()'s y-sign-flip quirk for
    pure-vertical actions (action[0] == 0) so the computed path matches
    what the environment will actually trace."""
    path = []
    pos = np.array(start, dtype=float)
    for a in actions:
        a = np.array(a, dtype=float)
        a_rev = np.array([0.0, -a[1]]) if a[0] == 0 else a
        norm = np.linalg.norm(a_rev)
        if norm > 0:
            a_rev = a_rev / norm
        pos = pos + step_size * a_rev
        path.append(tuple(pos))
    return path


def _nearest_state_xy(target, room_w, room_d, state_density=1):
    """Mirror Whittington2020._compute_reward_state_ids's nearest-state logic."""
    res_w, res_d = int(state_density * room_w), int(state_density * room_d)
    x_arr = np.linspace(-room_w / 2 + 0.5 / state_density, room_w / 2 - 0.5 / state_density, res_w)
    y_arr = np.linspace(-room_d / 2 + 0.5 / state_density, room_d / 2 - 0.5 / state_density, res_d)
    xy = np.stack(np.meshgrid(x_arr, y_arr), axis=-1).reshape(-1, 2)
    diff = np.sum((xy - np.array(target, dtype=float)) ** 2, axis=-1)
    return tuple(xy[np.argmin(diff)])


def _load_model_from_save(save_path):
    model_file = os.path.join(save_path, "whittington_2020_model.py")
    spec = importlib.util.spec_from_file_location("tem_model_saved_" + save_path, model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model


def build_loop_env_agent(save_path):
    training_dict = pd.read_pickle(os.path.join(save_path, "params.dict"))
    agent_params = dict(training_dict["agent_params"])
    env_params = dict(training_dict["env_params"])
    discrete_env_params = dict(env_params["arg_env_params"])
    state_dict = pd.read_pickle(os.path.join(save_path, "agent"))
    _load_model_from_save(save_path)

    room_w = agent_params["room_widths"][0]
    room_d = agent_params["room_depths"][0]
    reward_location = discrete_env_params["reward_location"]

    full_path = [LOOP_START] + _simulate_path(LOOP_START, SQUARE_LOOP, step_size=1)
    assert np.allclose(full_path[-1], full_path[0]), (
        f"SQUARE_LOOP is not closed: starts {full_path[0]}, ends {full_path[-1]}"
    )
    unique_path_states = full_path[:-1]

    reward_xy = _nearest_state_xy(reward_location, room_w, room_d)
    reward_idx = next(
        (i for i, p in enumerate(unique_path_states) if np.allclose(p, reward_xy)), None
    )
    assert reward_idx is not None, (
        f"Loop path {unique_path_states} does not reach reward-adjacent state {reward_xy} "
        f"given this checkpoint's env-0 room ({room_w}x{room_d}, reward_location={reward_location}) "
        f"- this checkpoint's arena/reward geometry doesn't match what the loop assumes."
    )
    approach_states = unique_path_states[: reward_idx + 1]
    n_landmarks = discrete_env_params.get("n_landmarks", 10)
    landmark_positions = approach_states[:n_landmarks]

    discrete_env_params["landmark_positions"] = landmark_positions
    env_params["arg_env_params"] = discrete_env_params

    env = training_dict["env_class"](**env_params)
    agent = training_dict["agent_class"](**agent_params)
    agent.tem.load_state_dict(state_dict)
    agent.tem.eval()

    return env, agent, unique_path_states, approach_states, landmark_positions


def run_probe(save_path, label):
    env, agent, unique_path_states, approach_states, landmark_positions = build_loop_env_agent(save_path)
    n_loop_states = len(unique_path_states)
    n_envs = agent.pars["batch_size"]
    n_rollout = agent.pars["n_rollout"]

    print(f"[{label}] square loop has {n_loop_states} unique states, "
          f"{len(approach_states)} on the approach to reward, "
          f"{len(landmark_positions)} explicit value-landmark positions", flush=True)

    random.seed(TRAJECTORY_SEED)
    np.random.seed(TRAJECTORY_SEED)
    obs, _ = env.reset(random_state=False, custom_state=list(LOOP_START))
    agent.n_walk = 0
    agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

    loop_idx = 0          # running counter, never wrapped
    prev_iter = None

    log_lap, log_pos, log_is_landmark, log_p0 = [], [], [], []

    with torch.no_grad():
        for chunk_i in range(N_CHUNKS):
            loop_idx_log = []
            while agent.n_walk < n_rollout:
                env0_state_pre = obs[0][0]
                loop_idx_log.append(loop_idx % n_loop_states)

                actions = agent.batch_act(obs)
                loop_action = SQUARE_LOOP[loop_idx % len(SQUARE_LOOP)]
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
            pos_at_step = [loop_idx_log[i] for i in real_indices]
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

            for i, step in enumerate(forward):
                pos = pos_at_step[i]
                lap = (chunk_i * n_rollout + i) // n_loop_states  # coarse, fine for logging
                p0_env0 = step.p_inf[0][0].detach().cpu().numpy()
                log_lap.append(lap)
                log_pos.append(pos)
                log_is_landmark.append(pos < len(landmark_positions))
                log_p0.append(p0_env0)

            prev_iter = [forward[-1]]
            agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

            completed_chunks = chunk_i + 1
            if completed_chunks % CONTROL_RESHUFFLE_EVERY_CHUNK == 0:
                for j in range(n_envs):
                    env.environments[j].reshuffle_control_landmarks()

    p0_mat = np.stack(log_p0, axis=0)
    lap_arr = np.array(log_lap)
    pos_arr = np.array(log_pos)
    is_landmark_arr = np.array(log_is_landmark)

    return lap_arr, pos_arr, is_landmark_arr, p0_mat, n_loop_states


def compute_backward_shift(lap_arr, pos_arr, is_landmark_arr, p0_mat, n_loop_states, label):
    """For each (lap, pos) row, find which loop position in lap 0's reference
    profile its p_inf vector best matches (cosine similarity), and report
    that match's circular offset from the row's actual position - the
    "shift" metric described at the top of this file."""
    lap0_mask = lap_arr == 0
    ref_profile = np.zeros((n_loop_states, p0_mat.shape[1]))
    for pos in range(n_loop_states):
        rows = p0_mat[lap0_mask & (pos_arr == pos)]
        if len(rows) > 0:
            ref_profile[pos] = rows.mean(axis=0)

    ref_norm = ref_profile / (np.linalg.norm(ref_profile, axis=1, keepdims=True) + 1e-9)
    p0_norm = p0_mat / (np.linalg.norm(p0_mat, axis=1, keepdims=True) + 1e-9)
    sims = p0_norm @ ref_norm.T  # (n_rows, n_loop_states)
    best_ref_pos = np.argmax(sims, axis=1)

    raw_shift = best_ref_pos - pos_arr
    shift = ((raw_shift + n_loop_states // 2) % n_loop_states) - n_loop_states // 2  # circular, centered at 0

    max_lap = lap_arr.max()
    rows = []
    for lap in range(max_lap + 1):
        lap_mask = lap_arr == lap
        for group_name, group_mask in [("landmark", is_landmark_arr), ("non_landmark", ~is_landmark_arr)]:
            mask = lap_mask & group_mask
            if mask.sum() == 0:
                continue
            rows.append({
                "label": label, "lap": lap, "group": group_name,
                "mean_shift": float(shift[mask].mean()), "n": int(mask.sum()),
            })
    df = pd.DataFrame(rows)

    print(f"\n[{label}] mean backward-shift by lap (negative = shifted earlier/backward):")
    for group_name in ["landmark", "non_landmark"]:
        sub = df[df["group"] == group_name]
        if len(sub) == 0:
            continue
        early = sub[sub["lap"] <= 2]["mean_shift"].mean()
        late = sub[sub["lap"] >= max_lap - 2]["mean_shift"].mean()
        print(f"  {group_name:14s} early(laps<=2)={early:+.3f}  late(laps>={max_lap-2})={late:+.3f}  "
              f"delta={late - early:+.3f}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    lap_arr, pos_arr, is_landmark_arr, p0_mat, n_loop_states = run_probe(args.save_path, args.label)
    df = compute_backward_shift(lap_arr, pos_arr, is_landmark_arr, p0_mat, n_loop_states, args.label)
    df.to_csv(args.out, index=False)
    print(f"Saved {args.out}")

    npz_path = os.path.splitext(args.out)[0] + "_raw.npz"
    np.savez(npz_path, lap=lap_arr, pos=pos_arr, is_landmark=is_landmark_arr, p0=p0_mat)
    print(f"Saved {npz_path}")


if __name__ == "__main__":
    main()
