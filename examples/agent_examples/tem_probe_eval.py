"""Probe evaluation: frozen-weight walk -> rate maps -> predictive analysis.

Two-phase walk for env 0:
  Phase 1 (N_RANDOM_STEPS):   random policy — warms up obs_history, lets M settle
  Phase 2 (N_CIRCULAR_STEPS): circular policy — repeated reward visits on a smooth
                               arc through [0,0] and [3,3]; other 15 envs stay random

The circle is centred at the midpoint of start and reward with radius equal to
the distance from centre to either endpoint, so the arc passes exactly through
both [0,0] and [3,3].  Waypoints are evenly spaced around the circle; at each
step the agent takes the grid action (up/down/left/right) most aligned with the
direction to the next waypoint.  Because the target direction rotates continuously
around the circle, the agent curves naturally instead of stair-stepping.

run_eval uses the last EVAL_STEPS=500 steps (all from phase 2) for rate maps.

Follows the notebook loading pattern:
  - agent_hyper loaded from disk
  - params.dict loaded from disk
  - model loaded from the saved whittington_2020_model.py copy in results/

Usage
-----
    cd examples/agent_examples
    python tem_probe_eval.py
"""

import importlib.util
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from _tem_eval import run_eval

# ── Config ────────────────────────────────────────────────────────────────────
TRAJECTORY_SEED   = 42
EPISODE_LABEL     = 10000
N_RANDOM_STEPS    = 5000    # phase 1: random policy
N_CIRCULAR_STEPS  = 5000    # phase 2: square-loop policy
# Use actual grid state centres, not the nominal [0,0]/[3,3] coordinates.
# custom_state=[0,0] snaps to nearest centre (-0.5,-0.5); reward [3,3] snaps to (2.5,2.5).
START_POS         = [0, 0]   # snaps to nearest grid state centre (-0.5, -0.5)
# ──────────────────────────────────────────────────────────────────────────────

# Fixed square-loop action sequence: 3× up, 3× right, 3× down, 3× left, repeat.
# Applied identically to ALL environments in phase 2.
_UP    = [0,  1]
_DOWN  = [0, -1]
_RIGHT = [1,  0]
_LEFT  = [-1, 0]
SQUARE_LOOP = (
    [_DOWN]  * 4 +
    [_RIGHT] * 3 +
    [_UP]    * 4 +
    [_LEFT]  * 3
)


def _load_model_from_save(save_path):
    """Load TEM Model class from the copy saved alongside the weights."""
    model_file = os.path.join(save_path, "whittington_2020_model.py")
    spec = importlib.util.spec_from_file_location("tem_model_saved", model_file)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model


def probe_condition(condition: str):
    use_reward = (condition == "reward_modulated")
    save_path  = os.path.join(os.getcwd(), "results_sim", condition)
    agent_path = os.path.join(save_path, "agent")

    if not os.path.exists(agent_path):
        print(f"[{condition}] No saved agent at {save_path} — skipping.")
        return

    print(f"\n{'='*60}")
    print(f"Probing: {condition}")
    print(f"{'='*60}")

    # ── Load saved metadata ───────────────────────────────────────────────────
    training_dict = pd.read_pickle(os.path.join(save_path, "params.dict"))
    agent_params  = training_dict["agent_params"]
    hyper         = pd.read_pickle(os.path.join(save_path, "agent_hyper"))
    state_dict    = pd.read_pickle(agent_path)

    # Rebuild TEM from saved model file and load trained weights
    ModelClass = _load_model_from_save(save_path)
    tem = ModelClass(hyper)
    tem.load_state_dict(state_dict)
    tem.eval()
    print(f"  TEM weights loaded ({len(state_dict)} tensors)")

    # ── Rebuild environment and agent ─────────────────────────────────────────
    env   = training_dict["env_class"](**training_dict["env_params"])
    agent = training_dict["agent_class"](**agent_params)
    agent.tem.load_state_dict(state_dict)
    agent.tem.eval()

    # ── Seed and reset ────────────────────────────────────────────────────────
    random.seed(TRAJECTORY_SEED)
    np.random.seed(TRAJECTORY_SEED)
    obs, state = env.reset(random_state=False, custom_state=START_POS)

    n_rollout    = agent.pars["n_rollout"]
    n_envs       = len(obs)
    total_steps  = N_RANDOM_STEPS + N_CIRCULAR_STEPS
    steps_taken  = 0
    episode      = 0
    loop_idx     = 0   # advances each committed step in phase 2

    print(f"  Phase 1: {N_RANDOM_STEPS} random steps...")
    print(f"  Phase 2: {N_CIRCULAR_STEPS} square-loop steps (all envs)...")

    while steps_taken < total_steps:
        n_walk_before = agent.n_walk

        actions = agent.batch_act(obs)

        # Phase 2: overwrite every env's action with the current square-loop move.
        # loop_idx advances only when a step is committed so the sequence is
        # consumed at exactly the rate the agent actually moves.
        if steps_taken >= N_RANDOM_STEPS:
            loop_action = SQUARE_LOOP[loop_idx % len(SQUARE_LOOP)]
            for i in range(n_envs):
                actions[i] = loop_action
                agent.prev_actions[i] = loop_action

        obs, state, reward = env.step(actions, normalize_step=True)

        if agent.n_walk > n_walk_before:
            steps_taken += 1
            if steps_taken >= N_RANDOM_STEPS:
                loop_idx += 1
            if steps_taken == N_RANDOM_STEPS:
                print(f"  >> Switching to square-loop policy at step {steps_taken}",
                      flush=True)
                obs, _ = env.reset(random_state=False, custom_state=[-0.5, -0.5])

        # Reset n_walk after each rollout so the collection loop continues
        if agent.n_walk >= n_rollout:
            agent.n_walk = 0
            if agent.use_reward:
                agent.td_errors = agent.td_errors[n_rollout:]
            episode += 1
            if episode % 50 == 0:
                phase = "loop" if steps_taken >= N_RANDOM_STEPS else "random"
                print(f"    ep {episode:4d}  steps {steps_taken:5d}  [{phase}]",
                      flush=True)

    agent.tem.train()
    print(f"  Collected {steps_taken} steps across {episode} episodes.")
    print(f"  Running eval (last 500 steps = loop phase) -> episode_{EPISODE_LABEL}/")
    run_eval(agent, env, EPISODE_LABEL, save_path)
    print(f"  Done: {save_path}/plots/episode_{EPISODE_LABEL}/")


if __name__ == "__main__":
    probe_condition("baseline")
    probe_condition("reward_modulated")

    print("\nRunning predictive analysis...")
    import tem_predictive_analysis as pa
    pa.plot_population_activity_maps()
    pa.plot_value_correlation()
    pa.plot_peak_distance()
    pa.plot_grid_scores()
    pa.plot_proximal_cell_count()
    print(f"\nAll done. Results in: {pa.OUT_DIR}")
