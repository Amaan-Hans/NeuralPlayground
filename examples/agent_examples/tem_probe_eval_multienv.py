"""Multi-environment probe: frozen-weight random walk across all 16 envs ->
per-env rate maps -> reward-zone field enrichment analysis.

Unlike tem_probe_eval.py (which recovers a single env-0 checkpoint using a
square-loop trajectory), this script exists to get statistically usable field
counts for the reward-zone enrichment test in tem_predictive_analysis.py,
which needs pooled data across many environments - one 100-144-state
environment alone yields too few place fields for a meaningful shuffle test.

Runs a long *random*-policy walk (not the square loop) across all 16 envs
simultaneously so every environment's state space gets broad coverage, then
calls the (now multi-env-aware) run_eval() from _tem_eval.py to produce
p_rates_multienv.npz + env_meta.pkl for every environment in the batch.

Saved under a distinct episode label (20000) so it doesn't collide with the
existing training checkpoints (1000-5000) or the square-loop demo probe
(10000) from tem_probe_eval.py.

Usage
-----
    cd examples/agent_examples
    conda activate tem_env
    python tem_probe_eval_multienv.py
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
TRAJECTORY_SEED = 42
EPISODE_LABEL   = 20000
N_STEPS         = 30000   # random-policy steps; ~30000/144 ~ 200 visits/state
                           # for the largest (12x12) envs
START_POS       = [0, 0]
# ──────────────────────────────────────────────────────────────────────────────


def _load_model_from_save(save_path):
    """Load TEM Model class from the copy saved alongside the weights."""
    model_file = os.path.join(save_path, "whittington_2020_model.py")
    spec = importlib.util.spec_from_file_location("tem_model_saved", model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model


def probe_condition(condition: str):
    save_path = os.path.join(os.getcwd(), "results_sim", condition)
    agent_path = os.path.join(save_path, "agent")

    if not os.path.exists(agent_path):
        print(f"[{condition}] No saved agent at {save_path} — skipping.")
        return

    print(f"\n{'=' * 60}")
    print(f"Multi-env probing: {condition}")
    print(f"{'=' * 60}")

    # ── Load saved metadata ───────────────────────────────────────────────────
    training_dict = pd.read_pickle(os.path.join(save_path, "params.dict"))
    agent_params = training_dict["agent_params"]
    hyper = pd.read_pickle(os.path.join(save_path, "agent_hyper"))
    state_dict = pd.read_pickle(agent_path)

    # Rebuild TEM from saved model file and load trained weights
    ModelClass = _load_model_from_save(save_path)
    tem = ModelClass(hyper)
    tem.load_state_dict(state_dict)
    tem.eval()
    print(f"  TEM weights loaded ({len(state_dict)} tensors)")

    # ── Rebuild environment and agent ─────────────────────────────────────────
    env = training_dict["env_class"](**training_dict["env_params"])
    agent = training_dict["agent_class"](**agent_params)
    agent.tem.load_state_dict(state_dict)
    agent.tem.eval()

    # ── Seed and reset ────────────────────────────────────────────────────────
    random.seed(TRAJECTORY_SEED)
    np.random.seed(TRAJECTORY_SEED)
    obs, state = env.reset(random_state=False, custom_state=START_POS)

    n_rollout = agent.pars["n_rollout"]
    episode = 0
    steps_taken = 0

    print(f"  Random-policy walk: {N_STEPS} steps across all 16 envs...")

    while steps_taken < N_STEPS:
        n_walk_before = agent.n_walk
        actions = agent.batch_act(obs)
        obs, state, reward = env.step(actions, normalize_step=True)

        if agent.n_walk > n_walk_before:
            steps_taken += 1

        if agent.n_walk >= n_rollout:
            agent.n_walk = 0
            episode += 1
            if episode % 100 == 0:
                print(f"    ep {episode:5d}  steps {steps_taken:6d}", flush=True)

    agent.tem.train()
    print(f"  Collected {steps_taken} steps across {episode} episodes.")
    print(f"  Running multi-env eval -> episode_{EPISODE_LABEL}/")
    run_eval(agent, env, EPISODE_LABEL, save_path)
    print(f"  Done: {save_path}/plots/episode_{EPISODE_LABEL}/")


if __name__ == "__main__":
    probe_condition("baseline")
    probe_condition("reward_modulated")

    print("\nRunning reward-zone field enrichment analysis...")
    import tem_predictive_analysis as pa
    pa.plot_reward_zone_enrichment()
    print(f"\nAll done. Results in: {pa.OUT_DIR}")
