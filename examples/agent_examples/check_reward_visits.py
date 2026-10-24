"""Does the agent's walk actually reach the reward state often enough to
matter, at probe-scale (1000 episodes) and training-scale (50000 episodes)?

action_policy() is a UNIFORM RANDOM walk over 4 actions - value/reward never
bias which direction the agent moves (this TEM implementation learns value
passively via TD on top of random exploration; it does not do goal-directed
action selection). So how often the agent lands on the single reward_location
state - the only thing that ever sets r=1.0 for the TD update, and the state
the dedicated reward_object_id occupies - depends purely on walk length and
arena coverage, not on anything learned. This script measures it directly
rather than assuming it.

Builds env/agent from an existing checkpoint's params.dict (arena sizes,
reward_location, batch_size) but LOADS NO TRAINED WEIGHTS and calls
agent.batch_act()/env.step() only - never agent.update() - so no TEM forward
pass ever runs. This makes even a 50000*n_rollout-step walk fast (the cost is
pure environment/bookkeeping overhead, not neural net compute).

Usage: python check_reward_visits.py [params_dict_path] [--n-episodes N]
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

AGENT_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AGENT_EXAMPLES_DIR)
sys.path.insert(0, os.path.join(AGENT_EXAMPLES_DIR, "..", ".."))

DEFAULT_PARAMS_DICT = (
    r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis"
    r"\reward_modulated_experiments_ctrlshuffle25_rewardobj_lc\seed_42\baseline_lc\params.dict"
)
FLUSH_EVERY = 20000  # raw steps between tally-and-clear passes, to keep obs_history bounded


def count_visits(params_dict_path, n_episodes, trajectory_seed=42):
    random.seed(trajectory_seed)
    np.random.seed(trajectory_seed)

    training_dict = pd.read_pickle(params_dict_path)
    agent_params = dict(training_dict["agent_params"])
    env_params = dict(training_dict["env_params"])

    env = training_dict["env_class"](**env_params)
    agent = training_dict["agent_class"](**agent_params)

    n_envs = agent.batch_size
    n_rollout = agent.pars["n_rollout"]
    target_steps = n_episodes * n_rollout
    print(f"n_envs={n_envs} n_rollout={n_rollout} target_steps={target_steps} "
          f"(= {n_episodes} episodes x {n_rollout} rollout)")
    print(f"reward_state_ids (per env slot): {agent.reward_state_ids}")
    print(f"room sizes (states): {[int(w * d) for w, d in zip(agent.room_widths, agent.room_depths)]}")

    obs, state = env.reset(random_state=False, custom_state=[0, 0])

    visit_counts = np.zeros(n_envs, dtype=np.int64)
    total_real_steps = 0
    last_flushed = 0

    while total_real_steps < target_steps:
        actions = agent.batch_act(obs)
        obs, state, reward = env.step(actions, normalize_step=True)
        if len(agent.obs_history) - last_flushed > 0:
            pass  # obs_history grows by at most 1 per call; handled below
        if len(agent.obs_history) > last_flushed:
            entry = agent.obs_history[-1]
            for j in range(n_envs):
                if entry[j][0] == agent.reward_state_ids[j]:
                    visit_counts[j] += 1
            total_real_steps += 1
            last_flushed = len(agent.obs_history)
        if len(agent.obs_history) >= FLUSH_EVERY:
            agent.obs_history, agent.walk_actions = [], []
            agent.held_landmark_history = []
            last_flushed = 0

    return visit_counts, total_real_steps, n_rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_dict_path", nargs="?", default=DEFAULT_PARAMS_DICT)
    parser.add_argument("--n-episodes", type=int, nargs="+", default=[1000, 50000])
    args = parser.parse_args()

    for n_episodes in args.n_episodes:
        print(f"\n=== {n_episodes}-episode walk ===")
        visit_counts, total_real_steps, n_rollout = count_visits(args.params_dict_path, n_episodes)
        print(f"total real steps per env slot: {total_real_steps}")
        print(f"reward-state visits per env slot: min={visit_counts.min()} "
              f"mean={visit_counts.mean():.1f} max={visit_counts.max()}")
        print(f"fraction of steps AT the reward state: "
              f"mean={visit_counts.mean() / total_real_steps:.5f} "
              f"(1/n_states would be the naive uniform-coverage baseline)")
        zero_envs = int(np.sum(visit_counts == 0))
        print(f"env slots with ZERO reward-state visits: {zero_envs}/{len(visit_counts)}")


if __name__ == "__main__":
    main()
