"""Multi-environment probe: frozen-weight random walk across all 16 envs ->
per-env rate maps -> reward-zone field enrichment analysis.

Unlike tem_probe_eval.py (which recovers a single env-0 checkpoint using a
square-loop trajectory), this script exists to get statistically usable field
counts for the reward-zone enrichment test in tem_predictive_analysis.py,
which needs pooled data across many environments - one 100-144-state
environment alone yields too few place fields for a meaningful shuffle test.

Runs a single long *random*-policy walk of N_EPISODES rollouts across all 16
envs simultaneously via run_multienv_probe() in _tem_eval.py, which drives
the walk AND the rate-map forward pass together, one chunk (CHUNK_EPISODES
rollouts) at a time - each chunk's raw observation history is discarded
immediately after being folded into a running sum/count accumulator, so
memory stays flat regardless of N_EPISODES rather than growing for the whole
walk before being processed.

No env-0 diagnostic plots (place_cells_*.png, grid_cells_*.png,
trajectory.png, ...) - this script produces one rate-map array per
environment per condition, meant to be pooled and summarised by
tem_predictive_analysis.py, not browsed frame-by-frame.

Output is saved under results_sim/predictive_analysis/probe/ (its own
subfolder, not mixed into predictive_analysis/'s flat file list, and NOT
into results_sim/<condition>/plots/ - that folder is scanned by several
other analyses (population activity, peak distance, grid scores, proximal
cell count) that should only ever see real training checkpoints, and a
frozen-weight probe of the *final* trained model has nothing to do with any
specific training episode). A snapshot is written every SAVE_EVERY_EPISODES
episodes (overwriting the same file), so progress is visible on disk as the
probe runs and the run is resumable from the last snapshot if interrupted -
the final post-completion save is what tem_predictive_analysis.py treats as
authoritative.

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from _tem_eval import run_multienv_probe

# ── Config ────────────────────────────────────────────────────────────────────
TRAJECTORY_SEED     = 123
N_EPISODES          = 5000    # rollouts (not raw steps) - total steps = N_EPISODES * n_rollout
CHUNK_EPISODES      = 10      # rollouts processed per forward-pass chunk
WARMUP_STEPS        = 500     # dropped from the start of the window before averaging
SAVE_EVERY_EPISODES = 500     # write a snapshot this often (overwrites previous snapshot)
START_POS           = [0, 0]

RESULTS_DIR   = os.path.join(os.getcwd(), "results_sim")
PROBE_OUT_DIR = os.path.join(RESULTS_DIR, "predictive_analysis", "probe")
# ──────────────────────────────────────────────────────────────────────────────


def _load_model_from_save(save_path):
    """Load TEM Model class from the copy saved alongside the weights."""
    model_file = os.path.join(save_path, "whittington_2020_model.py")
    spec = importlib.util.spec_from_file_location("tem_model_saved", model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model


def probe_condition(condition: str):
    import pandas as pd

    save_path = os.path.join(RESULTS_DIR, condition)
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

    print(f"  Random-policy walk: {N_EPISODES} episodes across all 16 envs, "
          f"chunk={CHUNK_EPISODES} episodes, saving to {PROBE_OUT_DIR}...", flush=True)

    rates, env_meta = run_multienv_probe(
        agent, env, obs, n_episodes=N_EPISODES, chunk_episodes=CHUNK_EPISODES,
        warmup_steps=WARMUP_STEPS, progress_prefix=f"[{condition}] ",
        save_dir=PROBE_OUT_DIR, save_every_episodes=SAVE_EVERY_EPISODES,
        condition=condition,
    )
    if not rates:
        print(f"  [{condition}] No usable data from probe walk.")
        return

    print(f"  Saved: {os.path.join(PROBE_OUT_DIR, f'probe_{condition}_rates.npz')}")
    print(f"  Saved: {os.path.join(PROBE_OUT_DIR, f'probe_{condition}_meta.pkl')}")


if __name__ == "__main__":
    probe_condition("baseline")
    probe_condition("reward_modulated")

    print("\nRunning reward-zone field enrichment analysis...")
    import tem_predictive_analysis as pa
    pa.plot_reward_zone_enrichment()
    print(f"\nAll done. Results in: {pa.OUT_DIR}")
