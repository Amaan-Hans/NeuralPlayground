"""Like check_value_pullback.py, but runs the FULL 500 episodes through the
real episode-250 shift (reusing apply_shift from probe_loop_ring.py) and
snapshots V twice: once right at the pre/post boundary (episode 250, before
the shift lands), once at the end (episode 500) after the agent has actually
had 250 more episodes to re-learn value under the new landmark layout.

This is the genuine post-shift value table - NOT the pre-shift V simply
repainted onto the new landmark positions (which is what
plot_loop_rate_maps_2d.py was doing before this script existed: it reused
check_value_pullback.py's pre-shift-only V for both panels, since V is keyed
by landmark id and the shift keeps the same 5 ids - a reasonable "does value
transfer instantly via landmark identity" view, but not the same as showing
how much further value actually moves after real post-shift experience).

Saves V_pre and V_post (per landmark id) to value_pullback_full.csv in
Thesis/quantitative/loop/ for plot_loop_rate_maps_2d.py to read, and prints
both arrays.
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
    BATCH_SIZE, RING_LOOP, LANDMARK_DISTANCES, TRAJECTORY_SEED, SHIFT_AT_EPISODE, N_EPISODES_TOTAL,
)

THESIS = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis"
LOOP_DIR = os.path.join(THESIS, "quantitative", "loop")
SEED = 42
COND = "rm"
SAVE_PATH = os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj", f"seed_{SEED}", "reward_modulated")


def run_value_buildup_full(save_path, label):
    cfg = build_ring_env_agent(save_path)
    env, agent = cfg["env"], cfg["agent"]
    n_rollout = agent.pars["n_rollout"]

    assert agent.use_reward and agent.td is not None, f"{label}: agent has no TD value head"

    random.seed(TRAJECTORY_SEED)
    np.random.seed(TRAJECTORY_SEED)
    obs, state = env.reset(random_state=False, custom_state=list(cfg["unique_path_states"][0]))
    control_states = pick_control_states(cfg["non_value_states"], random)
    place_ring_objects(env, cfg, control_states)
    obs = [env.environments[j].make_object_observation(np.array(cfg["unique_path_states"][0])) for j in range(BATCH_SIZE)]
    agent.n_walk = 0
    agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

    loop_idx = 0
    v_pre = None
    with torch.no_grad():
        for episode in range(N_EPISODES_TOTAL):
            while agent.n_walk < n_rollout:
                agent.batch_act(obs)
                loop_action = RING_LOOP[loop_idx % len(RING_LOOP)]
                env0_state_pre = obs[0][0]
                for i in range(BATCH_SIZE):
                    agent.prev_actions[i] = loop_action
                obs, state, reward = env.step([loop_action] * BATCH_SIZE, normalize_step=True)
                if obs[0][0] != env0_state_pre:
                    loop_idx += 1
            agent.n_walk = 0
            agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

            if episode + 1 == SHIFT_AT_EPISODE:
                v_pre = np.mean(np.stack([agent.td.V[j][:len(LANDMARK_DISTANCES)] for j in range(BATCH_SIZE)]), axis=0)
                print(f"[{label}] V_pre (end of episode {episode + 1}, just before shift): {v_pre}", flush=True)
                # Matches run_probe() exactly: apply_shift() only rewrites
                # env_j.objects (+ reward_state_ids) in place - no manual obs
                # refresh needed, since the next env.step() call recomputes
                # each env's observation from its actual current state using
                # the just-updated object array.
                apply_shift(env, agent, cfg)

            if (episode + 1) % 50 == 0:
                print(f"  [{label}] episode {episode + 1}/{N_EPISODES_TOTAL}", flush=True)

    v_post = np.mean(np.stack([agent.td.V[j][:len(LANDMARK_DISTANCES)] for j in range(BATCH_SIZE)]), axis=0)
    print(f"[{label}] V_post (end of episode {N_EPISODES_TOTAL}): {v_post}")
    return v_pre, v_post


def main():
    v_pre, v_post = run_value_buildup_full(SAVE_PATH, COND)
    out = pd.DataFrame({
        "landmark_id": range(len(LANDMARK_DISTANCES)),
        "ring_states_from_reward": LANDMARK_DISTANCES,
        "V_pre": v_pre,
        "V_post": v_post,
    })
    out_path = os.path.join(LOOP_DIR, "value_pullback_full.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
