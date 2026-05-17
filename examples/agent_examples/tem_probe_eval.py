"""Probe evaluation: generate p_rates.npy and v_table.npy from saved weights.

Loads the trained TEM weights (state_dict) for both conditions, runs a short
frozen walk (~600 steps), then calls run_eval to produce rate maps and save
the .npy arrays needed by tem_predictive_analysis.py.

This is needed when training completed before the np.save calls were added
to _tem_eval.py.  Runs in ~5-10 minutes per condition instead of 6 hours.

Follows the same loading pattern as whittington_2020_example.ipynb:
  - agent_hyper loaded from disk (not re-derived from parameters.parameters())
  - params.dict loaded from disk (not hardcoded room dims)
  - model loaded from the saved copy of whittington_2020_model.py in results/

LIMITATION: only recovers the *endpoint* checkpoint (episode 10000).
Intermediate checkpoints (1000-9000) cannot be recovered without retraining.

Usage
-----
    cd examples/agent_examples
    python tem_probe_eval.py
"""

import importlib
import importlib.util
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from _tem_eval import run_eval

# ── Config ────────────────────────────────────────────────────────────────────
TRAJECTORY_SEED = 42
EPISODE_LABEL   = 10000   # label written into the output folder name
N_PROBE_STEPS   = 800     # steps to collect; run_eval uses the last 600
# ──────────────────────────────────────────────────────────────────────────────


def _load_model_from_save(save_path):
    """Load TEM Model class from the copy saved alongside the weights.

    Mirrors the notebook's importlib approach so the probe uses exactly
    the same architecture that was in place when training ran.
    """
    model_file = os.path.join(save_path, "whittington_2020_model.py")
    spec = importlib.util.spec_from_file_location("tem_model_saved", model_file)
    mod = importlib.util.module_from_spec(spec)
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

    # ── Load saved metadata (mirrors notebook cell 6) ────────────────────────
    training_dict = pd.read_pickle(os.path.join(save_path, "params.dict"))
    agent_params  = training_dict["agent_params"]   # room_widths, batch_size …
    env_params    = training_dict["env_class"], training_dict["env_params"]

    hyper = pd.read_pickle(os.path.join(save_path, "agent_hyper"))
    state_dict = pd.read_pickle(agent_path)

    # Rebuild TEM from the saved model file and load trained weights
    ModelClass = _load_model_from_save(save_path)
    tem = ModelClass(hyper)
    tem.load_state_dict(state_dict)
    tem.eval()
    print(f"  TEM weights loaded ({len(state_dict)} tensors)")

    # ── Rebuild environment (same params as training) ─────────────────────────
    env_class  = training_dict["env_class"]
    env        = env_class(**training_dict["env_params"])

    # ── Rebuild agent with the exact same constructor kwargs ──────────────────
    agent_class = training_dict["agent_class"]
    agent       = agent_class(**agent_params)

    # Swap in the trained TEM weights (agent.__init__ built a fresh Model)
    agent.tem.load_state_dict(state_dict)
    agent.tem.eval()

    # ── Reset env with the same seed used during training ─────────────────────
    random.seed(TRAJECTORY_SEED)
    np.random.seed(TRAJECTORY_SEED)
    obs, state = env.reset(random_state=False, custom_state=[0, 0])

    n_rollout   = agent.pars["n_rollout"]
    steps_taken = 0
    episode     = 0

    print(f"  Collecting {N_PROBE_STEPS} steps (weights frozen, no backprop)...")
    while steps_taken < N_PROBE_STEPS:
        while agent.n_walk < n_rollout:
            actions = agent.batch_act(obs)
            obs, state, reward = env.step(actions, normalize_step=True)
            steps_taken += 1
        # We only need obs_history filled; update() is not called so weights
        # never change.  Manually resetting n_walk lets the loop continue.
        agent.n_walk = 0
        if agent.use_reward:
            # keep td_errors buffer from growing unboundedly
            agent.td_errors = agent.td_errors[n_rollout:]
        episode += 1
        if episode % 10 == 0:
            print(f"    episode {episode:3d}  steps {steps_taken}", flush=True)

    agent.tem.train()

    print(f"  Collected {steps_taken} steps across {episode} episodes.")
    print(f"  Running eval -> episode_{EPISODE_LABEL}/")
    run_eval(agent, env, EPISODE_LABEL, save_path)
    print(f"  Done: {save_path}/plots/episode_{EPISODE_LABEL}/")


if __name__ == "__main__":
    probe_condition("baseline")
    probe_condition("reward_modulated")
    print("\nProbe complete. Now run: python tem_predictive_analysis.py")
