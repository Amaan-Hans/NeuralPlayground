"""Scripted square-loop probe with a mid-walk reward/landmark relocation:
walk the fixed SQUARE_LOOP (same geometry as whittington_2020_loop_run.py /
probe_loop_backward_shift.py) for 250 episodes with value landmarks +
reward placed explicitly along the approach-to-reward segment, then at
episode 250 shift the WHOLE explicit structure (every approach-path
landmark id AND the reward marker) back by exactly one loop-state together
(still on the loop, circularly - landmark id 0's old position, the loop
start, is vacated and backfilled as decoy/filler; the structure's "first"
position moves onto what used to be the return segment's last state), then
walk 250 more episodes. Frozen weights throughout (no agent.update()); the
scripted action sequence never changes, only the environment's object
layout and the agent's reward_state_ids (so TD still fires correctly at
the NEW reward location) are mutated at the shift boundary.

Tracks, separately for pre-shift (episodes 0-249) and post-shift (250-499):
  - per-episode object-prediction accuracy by TRUE category (landmark/
    control/reward/filler), for the loop-position-pooled "cycle plot"
    (period 14, matching the loop's own length, not the 25-episode
    control-reshuffle period used by the random-walk probes).
  - a 4x4 confusion matrix (true category x predicted category, predicted
    = argmax of the model's generative x_gt prediction), pooled over every
    step and env in that half.

Usage: python probe_loop_shift.py --save-path <checkpoint dir> --label <name> --out-dir <dir>
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
N_EPISODES_TOTAL = 500
SHIFT_AT_EPISODE = 250
CATEGORIES = ["landmark", "control", "reward", "filler"]

_DOWN, _RIGHT, _UP, _LEFT = [0, -1], [1, 0], [0, 1], [-1, 0]
SQUARE_LOOP = [_DOWN] * 4 + [_RIGHT] * 3 + [_UP] * 4 + [_LEFT] * 3
LOOP_START = (-0.5, -0.5)


def _simulate_path(start, actions, step_size=1):
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
    assert np.allclose(full_path[-1], full_path[0])
    unique_path_states = full_path[:-1]
    n_loop_states = len(unique_path_states)

    reward_xy = _nearest_state_xy(reward_location, room_w, room_d)
    reward_idx = next((i for i, p in enumerate(unique_path_states) if np.allclose(p, reward_xy)), None)
    assert reward_idx is not None, (
        f"Loop path does not reach reward-adjacent state {reward_xy} given room {room_w}x{room_d}."
    )
    approach_states = unique_path_states[: reward_idx + 1]
    n_landmarks = discrete_env_params.get("n_landmarks", 10)
    n_control_landmarks = discrete_env_params.get("n_control_landmarks", 0)
    landmark_positions = approach_states[:n_landmarks]

    discrete_env_params["landmark_positions"] = landmark_positions
    env_params["arg_env_params"] = discrete_env_params

    env = training_dict["env_class"](**env_params)
    agent = training_dict["agent_class"](**agent_params)
    agent.tem.load_state_dict(state_dict)
    agent.tem.eval()

    # Shifted structure: every explicit landmark/reward position moves back
    # by one loop-state (circularly, within the full 14-state loop) together.
    # landmark i currently sits at unique_path_states[i] (since
    # approach_states == unique_path_states[:reward_idx+1]); its shifted
    # position is one loop-index earlier, wrapping around the full loop.
    shifted_landmark_positions = [
        unique_path_states[(i - 1) % n_loop_states] for i in range(len(landmark_positions))
    ]
    new_reward_xy = shifted_landmark_positions[-1]

    return {
        "env": env, "agent": agent, "unique_path_states": unique_path_states,
        "approach_states": approach_states, "landmark_positions": landmark_positions,
        "shifted_landmark_positions": shifted_landmark_positions, "new_reward_xy": new_reward_xy,
        "reward_idx": reward_idx, "n_landmarks": n_landmarks, "n_control_landmarks": n_control_landmarks,
        "n_loop_states": n_loop_states,
    }


def apply_shift(env, agent, cfg):
    """Mutate every env slot's object layout: vacate the old explicit
    landmark/reward states (backfill as decoy/random, matching
    generate_objects()'s own fallback), write the new ones, and update the
    agent's reward_state_ids so TD(0) fires at the NEW reward location."""
    n_landmarks = cfg["n_landmarks"]
    n_special = n_landmarks + cfg["n_control_landmarks"]
    old_positions = cfg["landmark_positions"]
    new_positions = cfg["shifted_landmark_positions"]
    new_reward_xy = cfg["new_reward_xy"]

    for j in range(agent.pars["batch_size"]):
        env_j = env.environments[j]
        poss_objects = np.eye(env_j.n_objects)

        for xy in old_positions:
            state_id = int(env_j.pos_to_state(np.array(xy)))
            if env_j.decoy_object_id is not None:
                env_j.objects[state_id, :] = poss_objects[env_j.decoy_object_id]
            else:
                rand = random.randint(n_special, env_j.n_objects - 1)
                while rand == env_j.reward_object_id:
                    rand = random.randint(n_special, env_j.n_objects - 1)
                env_j.objects[state_id, :] = poss_objects[rand]

        for i, xy in enumerate(new_positions):
            state_id = int(env_j.pos_to_state(np.array(xy)))
            if i == len(new_positions) - 1 and env_j.reward_object_id is not None:
                env_j.objects[state_id, :] = poss_objects[env_j.reward_object_id]
            else:
                env_j.objects[state_id, :] = poss_objects[i]

        env_j.reward_location = list(new_reward_xy)
        agent.reward_state_ids[j] = int(env_j.pos_to_state(np.array(new_reward_xy)))


def categorize(obj_id, n_landmarks, n_special, reward_object_id):
    if reward_object_id is not None and obj_id == reward_object_id:
        return "reward"
    if obj_id < n_landmarks:
        return "landmark"
    if obj_id < n_special:
        return "control"
    return "filler"


def run_probe(save_path, label, out_dir):
    cfg = build_loop_env_agent(save_path)
    env, agent = cfg["env"], cfg["agent"]
    n_envs = agent.pars["batch_size"]
    n_rollout = agent.pars["n_rollout"]
    n_loop_states = cfg["n_loop_states"]
    n_landmarks = cfg["n_landmarks"]
    n_special = n_landmarks + cfg["n_control_landmarks"]
    reward_object_id = env.environments[0].reward_object_id

    print(f"[{label}] loop={n_loop_states} states, {len(cfg['approach_states'])} approach/landmark states, "
          f"shift at episode {SHIFT_AT_EPISODE}", flush=True)

    random.seed(TRAJECTORY_SEED)
    np.random.seed(TRAJECTORY_SEED)
    obs, state = env.reset(random_state=False, custom_state=list(LOOP_START))
    agent.n_walk = 0
    agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

    loop_idx = 0
    prev_iter = None
    rows = []
    confusion = {
        "pre": {t: {p: 0 for p in CATEGORIES} for t in CATEGORIES},
        "post": {t: {p: 0 for p in CATEGORIES} for t in CATEGORIES},
    }
    # Per-(half, loop position, true category) correct/total counts - this is
    # what the position-resolved "cycle plot" actually needs. Pooling by an
    # EPISODE's single starting position (as the per-episode rows below do)
    # is not the same thing: n_rollout=20 and n_loop_states=14 share a
    # factor of 2, so consecutive episodes' starting positions only ever
    # land on even indices - odd positions would never get their own point.
    pos_accum = {
        half: [{c: {"correct": 0, "total": 0} for c in CATEGORIES} for _ in range(n_loop_states)]
        for half in ("pre", "post")
    }

    with torch.no_grad():
        for episode in range(N_EPISODES_TOTAL):
            half = "pre" if episode < SHIFT_AT_EPISODE else "post"
            loop_idx_log = []
            while agent.n_walk < n_rollout:
                loop_idx_log.append(loop_idx % n_loop_states)
                actions = agent.batch_act(obs)
                loop_action = SQUARE_LOOP[loop_idx % len(SQUARE_LOOP)]
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

            cat_correct = {c: [] for c in CATEGORIES}
            for local_i, step in enumerate(forward):
                x_actual = step.x.detach().cpu().numpy()
                x_pred = step.x_gen[2].detach().cpu().numpy()
                true_ids = np.argmax(x_actual, axis=-1)
                pred_ids = np.argmax(x_pred, axis=-1)
                pos = pos_at_step[local_i]
                for j in range(n_envs):
                    true_cat = categorize(int(true_ids[j]), n_landmarks, n_special, reward_object_id)
                    pred_cat = categorize(int(pred_ids[j]), n_landmarks, n_special, reward_object_id)
                    correct = true_cat == pred_cat
                    cat_correct[true_cat].append(correct)
                    confusion[half][true_cat][pred_cat] += 1
                    pos_accum[half][pos][true_cat]["total"] += 1
                    pos_accum[half][pos][true_cat]["correct"] += int(correct)

            rows.append({
                "episode": episode, "half": half, "loop_pos_start": pos_at_step[0] if pos_at_step else -1,
                **{f"acc_{c}": (float(np.mean(cat_correct[c])) if cat_correct[c] else float("nan")) for c in CATEGORIES},
                **{f"n_{c}": len(cat_correct[c]) for c in CATEGORIES},
            })

            prev_iter = [forward[-1]]
            agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

            if episode + 1 == SHIFT_AT_EPISODE:
                print(f"[{label}]   >>> SHIFT at episode {episode + 1}: approach landmarks + reward "
                      f"move back 1 loop-state <<<", flush=True)
                apply_shift(env, agent, cfg)

            if (episode + 1) % 100 == 0:
                print(f"  [{label}] episode {episode + 1}/{N_EPISODES_TOTAL}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{label}_episodes.csv"), index=False)
    for half in ("pre", "post"):
        cm = pd.DataFrame(confusion[half]).T.reindex(index=CATEGORIES, columns=CATEGORIES)
        cm.to_csv(os.path.join(out_dir, f"{label}_confusion_{half}.csv"))

    pos_rows = []
    for half in ("pre", "post"):
        for pos in range(n_loop_states):
            for c in CATEGORIES:
                d = pos_accum[half][pos][c]
                pos_rows.append({
                    "half": half, "loop_pos": pos, "category": c,
                    "correct": d["correct"], "total": d["total"],
                    "accuracy": (d["correct"] / d["total"]) if d["total"] else float("nan"),
                })
    pd.DataFrame(pos_rows).to_csv(os.path.join(out_dir, f"{label}_position_accuracy.csv"), index=False)
    print(f"[{label}] saved episodes + confusion matrices to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    run_probe(args.save_path, args.label, args.out_dir)


if __name__ == "__main__":
    main()
