"""Two-phase training script for the Tolman-Eichenbaum Machine (TEM).

Phase 1 — Standard unsupervised TEM training
    The model navigates a batch of random 2-D arenas and learns grid-cell-like
    abstract location representations (entorhinal cortex) and place-cell-like
    grounded representations (hippocampus) entirely without reward.  This
    replicates the original Whittington et al. 2020 training procedure.

Phase 2 — TD-modulated Hebbian learning (dopamine experiment)
    A single reward location is introduced in each arena.  A linear value
    function V(s) = w_v · p_inf[0](s) is trained by TD(0) over the
    highest-frequency place cells.  The resulting TD prediction error delta
    is used to scale the Hebbian memory formation rate eta per step:

        eta_eff = eta_base * (1 + beta * max(delta, 0))

    This mimics phasic VTA dopamine gating hippocampal LTP: states that
    reliably predict reward have their place-cell memories consolidated more
    strongly, causing place fields to expand and shift backward along the
    approach trajectory to reward (temporal backward shift; Mehta et al. 2000;
    Stachenfeld et al. 2017).

Key parameters to adjust
    PHASE1_EPISODES : total TEM training iterations before reward is introduced.
                      Set to ~20000 for a fully converged run, 10 for a quick
                      smoke-test.
    PHASE2_EPISODES : iterations of TD-modulated training.
    TD_GAMMA        : discount factor (0.9 → ~10-step value horizon).
    TD_LR           : value function learning rate.
    TD_BETA         : Hebbian boost gain per unit positive TD error.
    REWARD_FRACTION : position of the reward state within each arena as a
                      fraction of n_states (0.0 = state 0, 1.0 = last state).
"""

import os
import pickle

import numpy as np
import torch

from neuralplayground.agents.whittington_2020 import Whittington2020
from neuralplayground.agents.whittington_2020_extras import (
    whittington_2020_parameters as parameters,
)
from neuralplayground.arenas import BatchEnvironment, DiscreteObjectEnvironment
from neuralplayground.backend import SingleSim, tem_training_loop
from neuralplayground.experiments import Sargolini2006Data

# =============================================================================
# Device selection
# =============================================================================
# Priority: CUDA GPU → Apple Silicon MPS → CPU.
# Override by passing device="cpu" (or "cuda:1" etc.) explicitly in
# agent_params below.
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# =============================================================================
# Experiment hyper-parameters
# =============================================================================

# --- Phase 1 (standard TEM) --------------------------------------------------
# Full training uses 20000; set to 10 for a quick smoke-test.
PHASE1_EPISODES = 10

# --- Phase 2 (TD dopamine modulation) ----------------------------------------
PHASE2_EPISODES = 2000   # iterations of reward-modulated Hebbian learning

# TD value function
TD_GAMMA = 0.9    # discount factor: 0.9 gives ~10-step effective horizon
TD_LR = 0.01      # value weight learning rate
TD_BETA = 3.0     # Hebbian gain: eta_eff = eta_base*(1 + beta*max(delta,0))
                  # beta=3 means a delta=1 triples the Hebbian rate that step

# Reward location: fraction of n_states along the linearised state index.
# 0.6 places the reward roughly 60% through the grid (not the very centre),
# making the backward shift visible over several steps before the reward.
REWARD_FRACTION = 0.6

# =============================================================================
# Environment and model setup (identical to original run.py)
# =============================================================================

simulation_id = "TEM_custom_sim"
save_path = os.path.join(os.getcwd(), "results_sim")

params = parameters.parameters()
full_agent_params = params.copy()

# 16 arenas of varying sizes — same as the original setup
arena_x_limits = [
    [-5, 5], [-4, 4], [-5, 5], [-6, 6], [-4, 4], [-5, 5], [-6, 6], [-5, 5],
    [-4, 4], [-5, 5], [-6, 6], [-5, 5], [-4, 4], [-5, 5], [-6, 6], [-5, 5],
]
arena_y_limits = [
    [-5, 5], [-4, 4], [-5, 5], [-6, 6], [-4, 4], [-5, 5], [-6, 6], [-5, 5],
    [-4, 4], [-5, 5], [-6, 6], [-5, 5], [-4, 4], [-5, 5], [-6, 6], [-5, 5],
]

room_widths = [int(np.diff(arena_x_limits)[i]) for i in range(len(arena_x_limits))]
room_depths = [int(np.diff(arena_y_limits)[i]) for i in range(len(arena_y_limits))]

discrete_env_params = {
    "environment_name": "DiscreteObject",
    "state_density": 1,
    "n_objects": params["n_x"],
    "agent_step_size": 1,
    "use_behavioural_data": False,
    "data_path": None,
    "experiment_class": Sargolini2006Data,
}

env_params = {
    "environment_name": "BatchEnvironment",
    "batch_size": 16,
    "arena_x_limits": arena_x_limits,
    "arena_y_limits": arena_y_limits,
    "env_class": DiscreteObjectEnvironment,
    "arg_env_params": discrete_env_params,
}

agent_params = {
    "model_name": "Whittington2020",
    "params": full_agent_params,
    "batch_size": env_params["batch_size"],
    "room_widths": room_widths,
    "room_depths": room_depths,
    "state_densities": [discrete_env_params["state_density"]] * env_params["batch_size"],
    "use_behavioural_data": False,
    # GPU support: passed through to Whittington2020.__init__ → Model.__init__.
    # Remove or set to "cpu" to force CPU regardless of hardware.
    "device": DEVICE,
}

# =============================================================================
# Phase 1: standard unsupervised TEM training
# =============================================================================
# We use SingleSim exactly as before.  The TD state attributes added to
# Whittington2020.__init__ are all None / False during Phase 1, so the model
# is bit-for-bit identical to the original.
print("=" * 60)
print("Phase 1: standard TEM training")
print(f"  Episodes: {PHASE1_EPISODES}")
print("=" * 60)

training_loop_params_p1 = {"n_episode": PHASE1_EPISODES, "params": full_agent_params}

sim = SingleSim(
    simulation_id=simulation_id,
    agent_class=Whittington2020,
    agent_params=agent_params,
    env_class=BatchEnvironment,
    env_params=env_params,
    training_loop=tem_training_loop,
    training_loop_params=training_loop_params_p1,
)

print("Running Phase 1 sim...")
sim.run_sim(save_path)
print("Phase 1 complete.\n")

# Save the Phase 1 agent so we can reload it later without re-training.
phase1_save = os.path.join(save_path, "phase1_agent.pkl")
os.makedirs(save_path, exist_ok=True)
with open(phase1_save, "wb") as f:
    pickle.dump(sim.agent.tem.state_dict(), f, pickle.HIGHEST_PROTOCOL)
print(f"Phase 1 model saved to: {phase1_save}")

# =============================================================================
# Phase 2: TD-modulated Hebbian learning (dopamine experiment)
# =============================================================================
# Reuse the already-trained agent and environment from Phase 1 rather than
# constructing them from scratch.  We then call activate_td_learning() to
# attach the value head and reward locations before running more episodes.
print("=" * 60)
print("Phase 2: TD-dopamine modulated Hebbian learning")
print(f"  Episodes   : {PHASE2_EPISODES}")
print(f"  TD gamma   : {TD_GAMMA}")
print(f"  TD lr      : {TD_LR}")
print(f"  TD beta    : {TD_BETA}")
print(f"  Reward pos : {REWARD_FRACTION:.0%} through each arena")
print("=" * 60)

# Retrieve the trained agent and environment from the Phase 1 simulation
agent = sim.agent
env = sim.env

# --- Define one reward location per environment ----------------------------
# We place reward at REWARD_FRACTION * n_states for each arena.
# Using the same fraction across arenas means the reward is at a comparable
# spatial position regardless of arena size, which keeps comparisons clean.
n_states = agent.n_states  # list[int], one per environment
reward_ids = [
    min(int(n * REWARD_FRACTION), n - 1)  # clamp to valid state range
    for n in n_states
]
print(f"Reward state IDs: {reward_ids}\n")

# Switch the agent into Phase 2 mode.
# This attaches a TDValueHead and initialises prev_eta_scales = None.
# The first Phase 2 update() call will run forward with no eta modulation
# (prev_eta_scales is None), then compute eta_scales from the TD errors of
# that rollout — which are stored and used from the second call onward.
agent.activate_td_learning(
    reward_ids=reward_ids,
    gamma=TD_GAMMA,
    lr=TD_LR,
    beta=TD_BETA,
)

# --- Phase 2 training loop -------------------------------------------------
# We call tem_training_loop directly rather than wrapping in SingleSim so
# that we can inspect the agent's value_head.w_v and prev_eta_scales during
# training without going through the sim abstraction.
print("Running Phase 2 training...")
agent, env, training_dict_p2 = tem_training_loop(
    agent=agent,
    env=env,
    n_episode=PHASE2_EPISODES,
    params=full_agent_params,
)
print("Phase 2 complete.\n")

# --- Save Phase 2 results --------------------------------------------------
phase2_save = os.path.join(save_path, "phase2_agent.pkl")
with open(phase2_save, "wb") as f:
    pickle.dump(agent.tem.state_dict(), f, pickle.HIGHEST_PROTOCOL)

# Also save the value function weights — these encode which states have
# accumulated reward-predictive value and can be inspected directly to
# verify that value backs up along the approach trajectory.
value_save = os.path.join(save_path, "phase2_value_weights.pkl")
with open(value_save, "wb") as f:
    pickle.dump(
        {
            "w_v": agent.value_head.w_v,           # [batch_size, n_p[0]]
            "reward_ids": reward_ids,
            "n_states": n_states,
            "td_gamma": TD_GAMMA,
            "td_beta": TD_BETA,
        },
        f,
        pickle.HIGHEST_PROTOCOL,
    )

print(f"Phase 2 TEM weights saved to : {phase2_save}")
print(f"Value function weights saved to: {value_save}")
print("\nDone.")
