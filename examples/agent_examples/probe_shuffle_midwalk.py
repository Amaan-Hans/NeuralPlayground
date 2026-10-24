"""1000-raw-step frozen-weight probe with a full object-layout reshuffle
at step 500: does one-step predictive accuracy crash and recover when the
environment changes out from under a frozen model, and does that differ
by condition (plain baseline, value only, LC gate only, value+LC)?

Weights are never updated (no agent.update() call, matching every other
probe script's convention). The TD value table is also never updated
during the probe (it's read-only here, same as accuracy_probe.py) - so
for the two value-carrying conditions, V(landmark_id) stays exactly as it
was at the end of training. After the reshuffle, a given landmark id may
now sit at a totally different (or no) position, so the agent's value
signal becomes STALE/mismatched to the new layout - this is deliberate,
not a bug: it's the sharpest version of "the world changed, does the
model's story about reward make sense anymore."

The reshuffle itself regenerates every environment's FULL object layout
(landmarks, controls, decoys, reward marker all re-drawn fresh via
DiscreteObjectEnvironment.generate_objects(), same reward_location/
landmark_bias_scale/decoy_object_id/reward_object_id as training) - a
bigger disruption than the existing reshuffle_control_landmarks()
mechanism, which only ever touches controls.

Accuracy is tracked per n_rollout-step chunk (20 steps/chunk here, so 25
chunks before the shuffle and 25 after, out of 1000 total steps) - fine
enough resolution to see immediate drop + any recovery trend.
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

MIXED_SIZE_CYCLE = [8.0, 9.0, 10.0, 11.0]
TRAJECTORY_SEED = 42
TOTAL_STEPS = 1000
SHUFFLE_AT_STEP = 500


def _load_model_from_save(save_path):
    model_file = os.path.join(save_path, "whittington_2020_model.py")
    spec = importlib.util.spec_from_file_location("tem_model_saved_" + save_path, model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model


def run_probe(save_path, label):
    training_dict = pd.read_pickle(os.path.join(save_path, "params.dict"))
    agent_params = dict(training_dict["agent_params"])
    env_params = dict(training_dict["env_params"])
    state_dict = pd.read_pickle(os.path.join(save_path, "agent"))
    _load_model_from_save(save_path)

    n_envs = agent_params["batch_size"]
    half_cycle = [[-s / 2, s / 2] for s in MIXED_SIZE_CYCLE]
    arena_x_limits = [half_cycle[i % len(half_cycle)] for i in range(n_envs)]
    arena_y_limits = [half_cycle[i % len(half_cycle)] for i in range(n_envs)]
    env_params["arena_x_limits"] = arena_x_limits
    env_params["arena_y_limits"] = arena_y_limits
    agent_params["room_widths"] = [s[1] - s[0] for s in arena_x_limits]
    agent_params["room_depths"] = [s[1] - s[0] for s in arena_y_limits]

    env = training_dict["env_class"](**env_params)
    agent = training_dict["agent_class"](**agent_params)
    agent.tem.load_state_dict(state_dict)
    agent.tem.eval()

    random.seed(TRAJECTORY_SEED)
    np.random.seed(TRAJECTORY_SEED)
    obs, state = env.reset(random_state=False, custom_state=[0, 0])
    n_rollout = agent.pars["n_rollout"]
    assert TOTAL_STEPS % n_rollout == 0 and SHUFFLE_AT_STEP % n_rollout == 0, \
        f"n_rollout={n_rollout} must divide both {TOTAL_STEPS} and {SHUFFLE_AT_STEP} evenly"
    n_chunks = TOTAL_STEPS // n_rollout
    shuffle_after_chunk = SHUFFLE_AT_STEP // n_rollout

    print(f"[{label}] n_envs={n_envs} n_rollout={n_rollout} n_chunks={n_chunks} "
          f"shuffle_after_chunk={shuffle_after_chunk}", flush=True)

    chunk_acc_p, chunk_acc_g, chunk_acc_gt = [], [], []
    prev_iter = None
    agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

    with torch.no_grad():
        for chunk_i in range(n_chunks):
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
                chunk_acc_p.append(float("nan"))
                chunk_acc_g.append(float("nan"))
                chunk_acc_gt.append(float("nan"))
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
            for step in forward:
                acc = step.correct()
                step_acc_p.append(acc[0])
                step_acc_g.append(acc[1])
                step_acc_gt.append(acc[2])
            chunk_acc_p.append(float(np.mean(np.stack(step_acc_p))))
            chunk_acc_g.append(float(np.mean(np.stack(step_acc_g))))
            chunk_acc_gt.append(float(np.mean(np.stack(step_acc_gt))))

            prev_iter = [forward[-1]]
            agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

            completed_chunks = chunk_i + 1
            if completed_chunks == shuffle_after_chunk:
                print(f"[{label}]   >>> RESHUFFLING all {n_envs} environments' object layouts "
                      f"after chunk {completed_chunks} ({completed_chunks * n_rollout} steps) <<<", flush=True)
                for j in range(n_envs):
                    env.environments[j].objects = env.environments[j].generate_objects()
                # obs currently held is stale (reflects the old layout at the
                # agent's unchanged position), but batch_act() only reads
                # obs[j][0] (location id, unchanged by this) to decide the
                # next action - it never reads the object encoding for
                # action selection. The very next env.step() call recomputes
                # the observation from the environment's (now reshuffled)
                # self.objects at the agent's current position, so the new
                # layout is correctly picked up from the next step onward
                # with no special handling needed here.

    rows = [{
        "label": label, "chunk": i, "step_start": i * n_rollout, "step_end": (i + 1) * n_rollout,
        "x_p": chunk_acc_p[i], "x_g": chunk_acc_g[i], "x_gt": chunk_acc_gt[i],
    } for i in range(n_chunks)]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = run_probe(args.save_path, args.label)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
