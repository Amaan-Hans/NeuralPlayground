"""Frozen-weight accuracy curve: all 5 seeds, 6 conditions total - the 4
rotation-enabled conditions (baseline, rm, baseline_lc, rm_lc) plus the 2
rotation-OFF conditions (baseline_noshuffle, rm_noshuffle, from the
ctrlshuffle25_rewardobj_noshuffle campaign - trained with control-landmark
reshuffling disabled entirely). Each walks 500 episodes (= 500 n_rollout-
step chunks = 10,000 raw steps), recording one-step predictive accuracy
(x_gt - did the model's generative prediction match the actual next
observation) per episode/chunk, broken out by the TRUE current object's
category (value landmark / control landmark / filler) - a 500-point time
series per condition/seed, not a single pooled number.

Rotation-enabled conditions get control-landmark reshuffling every
CONTROL_SHUFFLE_INTERVAL=25 episodes during the probe, matching the
ctrlshuffle25_rewardobj campaigns' own training-time interval. The
rotation-OFF conditions get no reshuffling at all during the probe either -
matching how they were actually trained, so "rotation off" is true start to
finish, not just at train time with an inconsistent probe.

Each checkpoint's own trained `use_reward` setting is left as-is (no value
on/off toggle here - this is the natural/as-trained behavior for every
condition). Weights are frozen throughout (torch.no_grad(), no
agent.update() call), matching every other probe script's convention.
"""
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

SEEDS = [42, 7, 1, 2, 3]
N_EPISODES = 500
CONTROL_SHUFFLE_INTERVAL = 25

# (root, condition subfolder, rotation_interval_or_None)
CAMPAIGNS = {
    "baseline": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj",
        "baseline", CONTROL_SHUFFLE_INTERVAL,
    ),
    "rm": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj",
        "reward_modulated", CONTROL_SHUFFLE_INTERVAL,
    ),
    "baseline_lc": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_lc",
        "baseline_lc", CONTROL_SHUFFLE_INTERVAL,
    ),
    "rm_lc": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_lc",
        "reward_modulated_lc", CONTROL_SHUFFLE_INTERVAL,
    ),
    "baseline_noshuffle": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle",
        "baseline", None,
    ),
    "rm_noshuffle": (
        r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle",
        "reward_modulated", None,
    ),
}

OUT_DIR = os.path.join(AGENT_EXAMPLES_DIR, "accuracy_curve_results")


def _load_model_from_save(save_path):
    import importlib.util
    model_file = os.path.join(save_path, "whittington_2020_model.py")
    spec = importlib.util.spec_from_file_location("tem_model_saved_" + save_path, model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model


def run_probe(save_path, label, seed, rotation_interval):
    training_dict = pd.read_pickle(os.path.join(save_path, "params.dict"))
    agent_params = dict(training_dict["agent_params"])
    state_dict = pd.read_pickle(os.path.join(save_path, "agent"))
    _load_model_from_save(save_path)

    env = training_dict["env_class"](**training_dict["env_params"])
    agent = training_dict["agent_class"](**agent_params)
    agent.tem.load_state_dict(state_dict)
    agent.tem.eval()

    n_envs = agent.pars["batch_size"]
    n_rollout = agent.pars["n_rollout"]
    n_landmarks = env.environments[0].n_landmarks
    n_control_landmarks = env.environments[0].n_control_landmarks
    n_special = n_landmarks + n_control_landmarks

    random.seed(seed)
    np.random.seed(seed)
    obs, state = env.reset(random_state=False, custom_state=[0, 0])
    agent.n_walk = 0
    agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

    print(f"[{label} seed={seed}] n_envs={n_envs} n_rollout={n_rollout} n_episodes={N_EPISODES} "
          f"control_shuffle_interval={rotation_interval}", flush=True)

    rows = []
    prev_iter = None

    with torch.no_grad():
        for episode in range(N_EPISODES):
            while agent.n_walk < n_rollout:
                actions = agent.batch_act(obs)
                obs, state, reward = env.step(actions, normalize_step=True)
            agent.n_walk = 0

            real_indices = [i for i, step in enumerate(agent.obs_history) if step[0][0] != -1]
            history = [agent.obs_history[i] for i in real_indices]
            actions_raw = [agent.walk_actions[i] for i in real_indices]
            held_raw = [agent.held_landmark_history[i] for i in real_indices]
            chunk_len = len(history)
            if chunk_len == 0:
                agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []
                rows.append({"episode": episode, "x_p": float("nan"), "x_g": float("nan"), "x_gt": float("nan")})
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

            step_acc_p, step_acc_g, step_acc_gt = [], [], []
            gt_landmark, gt_control, gt_filler = [], [], []
            for local_i, step in enumerate(forward):
                acc = step.correct()
                step_acc_p.append(acc[0])
                step_acc_g.append(acc[1])
                step_acc_gt.append(acc[2])
                gt_bool = acc[2]
                for j in range(n_envs):
                    obj_id = agent._object_id(history[local_i][j])
                    if obj_id is None:
                        continue
                    correct_here = bool(gt_bool[j])
                    if obj_id < n_landmarks:
                        gt_landmark.append(correct_here)
                    elif obj_id < n_special:
                        gt_control.append(correct_here)
                    else:
                        gt_filler.append(correct_here)
            rows.append({
                "episode": episode,
                "x_p": float(np.mean(np.stack(step_acc_p))),
                "x_g": float(np.mean(np.stack(step_acc_g))),
                "x_gt": float(np.mean(np.stack(step_acc_gt))),
                "x_gt_landmark": float(np.mean(gt_landmark)) if gt_landmark else float("nan"),
                "x_gt_control": float(np.mean(gt_control)) if gt_control else float("nan"),
                "x_gt_filler": float(np.mean(gt_filler)) if gt_filler else float("nan"),
                "n_landmark": len(gt_landmark), "n_control": len(gt_control), "n_filler": len(gt_filler),
            })

            prev_iter = [forward[-1]]
            agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

            completed = episode + 1
            if rotation_interval and completed % rotation_interval == 0:
                for j in range(n_envs):
                    env.environments[j].reshuffle_control_landmarks()

            if completed % 100 == 0:
                print(f"  [{label} seed={seed}] episode {completed}/{N_EPISODES}", flush=True)

    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for seed in SEEDS:
        for label, (root, condition, rotation_interval) in CAMPAIGNS.items():
            out_path = os.path.join(OUT_DIR, f"{label}_seed{seed}_accuracycurve.csv")
            if os.path.exists(out_path):
                print(f"skip (exists): {out_path}")
                continue
            save_path = os.path.join(root, f"seed_{seed}", condition)
            rows = run_probe(save_path, label, seed, rotation_interval)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"Saved {out_path}")
        print(f"### seed {seed} done ###", flush=True)


if __name__ == "__main__":
    main()
