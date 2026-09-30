"""Additive extension to _tem_eval.py: also save an all-environment,
place-cell-only CSV at each training checkpoint, alongside the existing
env-0-only plots run_eval() already produces (unchanged, still called as-is
- this module never modifies _tem_eval.py itself).

run_eval_combined() is the eval_fn to pass into tem_training_loop /
training_loop_params instead of plain run_eval - it calls run_eval()
first (so every existing plot/npy artifact is produced exactly as before),
then additionally computes and saves a per-env place-cell CSV using the
same trailing-window (EVAL_STEPS) live training history that run_eval()
already reads from.

IMPORTANT: this does NOT call _tem_eval.compute_multienv_rates() directly -
that function does a single monolithic forward pass over the entire
trailing window (up to EVAL_STEPS=500 steps x 16 envs) in one shot, which
CUDA-OOMs when both training subprocesses (baseline + reward_modulated)
hit their eval checkpoint around the same moment and share one GPU (this
was observed directly: 'agent_hyper' loads fine, training runs fine, but
the eval call itself dies with torch.OutOfMemoryError inside hebbian()'s M
update - because the Hebbian memory tensor gets retained across all 500
steps of one single forward() call, matching the exact failure mode
run_multienv_probe()'s docstring already warns about for the standalone
probe). _compute_multienv_rates_chunked() below reprocesses the SAME
trailing window in small chunks instead, carrying recurrent state across
chunk boundaries via prev_iter (this is the same technique
run_multienv_probe() already uses safely) - numerically identical result,
bounded peak GPU memory regardless of window size.

CSV column schema matches the standalone all-envs probe dataset
(tem_probe_eval_allenvs.py) exactly, with an added 'episode' column:
    episode, condition, env_id, state_id, x, y, is_landmark, is_reward,
    value, p_<freq>_<idx>...
"""

import os

import numpy as np
import pandas as pd
import torch

from _tem_eval import FREQ_NAMES, EVAL_STEPS, run_eval

# Small chunk size: two conditions (baseline, reward_modulated) run as
# concurrent subprocesses sharing one GPU throughout training, so eval calls
# here compete for memory with each other AND with the other subprocess's
# ongoing training step - keep this well under run_multienv_probe()'s
# already-proven-safe 200-steps-per-chunk (used when a probe has the whole
# GPU to itself).
CHUNK_SIZE = 50


def _compute_multienv_rates_chunked(agent, env, window_steps=EVAL_STEPS, chunk_size=CHUNK_SIZE):
    """Same computation and return shape as _tem_eval.compute_multienv_rates()
    - reprocesses the trailing `window_steps` of agent.obs_history for every
    environment, returning the mean of the second half of each state's
    visits within the window (matches run_eval()'s single-env convention) -
    but forward-passes it in `chunk_size`-step pieces instead of one call,
    to keep peak GPU memory bounded. See module docstring for why.
    """
    n_hist = len(agent.obs_history)
    if n_hist == 0:
        return None, None

    real_indices = [i for i, step in enumerate(agent.obs_history) if step[0][0] != -1]
    real_history = [agent.obs_history[i] for i in real_indices]
    real_actions = agent.walk_actions[-len(real_history):]

    n_steps = min(window_steps, len(real_history))
    if n_steps == 0:
        return None, None
    history_slice = real_history[-n_steps:]
    held_indices = real_indices[-n_steps:]
    walk_slice = real_actions[-n_steps:]

    n_envs = agent.batch_size
    n_f = agent.pars["n_f"]
    n_p_list = agent.pars["n_p"]
    n_states_list = agent.n_states

    p_accum = [
        [[[] for _ in range(n_states_list[j])] for _ in range(n_f)]
        for j in range(n_envs)
    ]

    action_values = agent.step_to_actions(walk_slice)
    action_array = np.reshape(action_values, (n_steps, n_envs))

    saved_batch_size = agent.tem.hyper.get("batch_size", n_envs)
    agent.tem.eval()
    prev_iter = None
    with torch.no_grad():
        for chunk_start in range(0, n_steps, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_steps)
            chunk_history = history_slice[chunk_start:chunk_end]
            chunk_held_idx = held_indices[chunk_start:chunk_end]
            chunk_len = len(chunk_history)

            locations_seq = [
                [{"id": step[j][0], "shiny": None} for j in range(n_envs)]
                for step in chunk_history
            ]
            obs_seq = np.array(
                [[step[j][1] for j in range(n_envs)] for step in chunk_history],
                dtype=np.float32,
            )

            v_seq = None
            if agent.use_reward and agent.td is not None:
                v_seq = np.zeros((chunk_len, n_envs), dtype=np.float32)
                for j in range(n_envs):
                    v_table = agent.td.V[j]
                    v_max = float(np.max(v_table))
                    for i, idx in enumerate(chunk_held_idx):
                        key = agent.held_landmark_history[idx][j]
                        if key is None:
                            continue
                        v_t = float(v_table[key]) if 0 <= key < v_table.shape[0] else 0.0
                        v_seq[i, j] = v_t / v_max if v_max > 0 else 0.0

            model_input = [
                [
                    locations_seq[i],
                    torch.tensor(obs_seq[i], dtype=torch.float32).to(agent.device),
                    [int(a) for a in action_array[chunk_start + i]],
                ]
                for i in range(chunk_len)
            ]
            if v_seq is not None:
                for i in range(chunk_len):
                    model_input[i].append(None)  # td_scale slot (Hebbian gating) - unused
                    model_input[i].append(
                        torch.tensor(v_seq[i], dtype=torch.float32).to(agent.device)
                    )

            forward = agent.tem(model_input, prev_iter=prev_iter)

            for step in forward:
                for j in range(n_envs):
                    loc = step.g[j]["id"]
                    if loc < 0:
                        continue
                    for f in range(n_f):
                        p_accum[j][f][loc].append(step.p_inf[f][j].detach().cpu().numpy())

            prev_iter = [forward[-1]]

    agent.tem.train()
    agent.tem.hyper["batch_size"] = saved_batch_size

    rates_to_save = {}
    for j in range(n_envs):
        freq_rates_all = []
        for f in range(n_f):
            freq_rates = []
            for loc in range(n_states_list[j]):
                visits = p_accum[j][f][loc]
                if not visits:
                    freq_rates.append(np.zeros(n_p_list[f]))
                else:
                    half = len(visits) // 2
                    usable = visits[half:] if half < len(visits) else visits
                    freq_rates.append(np.mean(usable, axis=0))
            freq_rates_all.append(np.stack(freq_rates, axis=0))
        rates_to_save[f"env{j}"] = np.concatenate(freq_rates_all, axis=1)

    env_meta = {
        "room_widths": list(agent.room_widths),
        "room_depths": list(agent.room_depths),
        "state_densities": list(agent.state_densities),
        "reward_location": agent.reward_location,
    }
    return rates_to_save, env_meta

# Optionally narrows which checkpoints ALSO get the heavier all-envs CSV
# (run_eval's own env-0 plots still fire at every checkpoint, including
# episode 1, regardless). None (default) = every checkpoint run_eval_combined
# is called at gets the CSV too - safe now that the computation is chunked.
# Set to an explicit set (e.g. {1000, 2000, 3000, 4000, 5000}) to restrict it,
# useful if checkpoints are frequent and every one is more than you need.
ALLENVS_CHECKPOINTS = None


def _col_names(n_p_list):
    names = []
    for f, freq in enumerate(FREQ_NAMES):
        for i in range(n_p_list[f]):
            names.append(f"p_{freq}_{i:03d}")
    return names


def _build_allenvs_df(agent, env, episode):
    condition = "reward_modulated" if agent.use_reward else "baseline"

    rates_to_save, env_meta = _compute_multienv_rates_chunked(agent, env)
    if rates_to_save is None:
        return None

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

        object_layout = env.environments[j].objects
        object_ids_per_state = np.argmax(object_layout, axis=1)
        # is_landmark = VALUE landmark only (tracked by held_landmark/TD).
        # is_control_landmark = the new never-duplicated, reward-biased
        # objects that sit outside agent.n_landmarks - sensorily identical
        # in kind, but never touched by the value mechanism. n_control
        # defaults to 0 (attribute may not exist on older-format envs).
        n_control = getattr(env.environments[j], "n_control_landmarks", 0)
        is_landmark = (object_ids_per_state < agent.n_landmarks).astype(int)
        is_control_landmark = (
            (object_ids_per_state >= agent.n_landmarks)
            & (object_ids_per_state < agent.n_landmarks + n_control)
        ).astype(int)

        v_per_state = np.full(n_states_j, np.nan, dtype=np.float32)
        if agent.use_reward and agent.td is not None:
            v_table = agent.td.V[j]
            for sid in range(n_states_j):
                obj_id = int(object_ids_per_state[sid])
                if obj_id < agent.n_landmarks:
                    v_per_state[sid] = v_table[obj_id]

        p_all = rates_to_save[f"env{j}"]  # (n_states_j, total_p_cells)
        xs, ys = zip(*[state_xy(s) for s in range(n_states_j)])

        meta_df = pd.DataFrame({
            "episode": episode,
            "condition": condition,
            "env_id": j,
            "state_id": np.arange(n_states_j),
            "x": xs,
            "y": ys,
            "is_landmark": is_landmark,
            "is_control_landmark": is_control_landmark,
            "is_reward": (np.arange(n_states_j) == reward_state_id).astype(int),
            "value": v_per_state,
        })
        feat_df = pd.DataFrame(p_all, columns=_col_names(n_p_list))
        rows.append(pd.concat([meta_df, feat_df], axis=1))

    return pd.concat(rows, ignore_index=True)


def run_eval_combined(agent, env, episode: int, eval_save_path: str):
    """Drop-in replacement for run_eval as an eval_fn: produces every
    existing env-0 artifact exactly as before, then additionally saves an
    all-environment place-cell CSV at the checkpoints in ALLENVS_CHECKPOINTS.
    """
    run_eval(agent, env, episode, eval_save_path)

    if ALLENVS_CHECKPOINTS is not None and episode not in ALLENVS_CHECKPOINTS:
        return

    df = _build_allenvs_df(agent, env, episode)
    if df is None:
        return

    ep_dir = os.path.join(eval_save_path, "plots", f"episode_{episode}")
    os.makedirs(ep_dir, exist_ok=True)
    out_path = os.path.join(ep_dir, "allenvs_place_cells.csv")
    df.to_csv(out_path, index=False)
    print(f"  [eval-allenvs ep {episode}] saved -> {out_path}", flush=True)
