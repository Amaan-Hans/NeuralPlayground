"""Training simulation for the Whittington et al.

2020 agent, the Tolman-Eichenbaum Machine (TEM). The TEM is a model of
the hippocampus that learns to navigate a series of environments and
solve a series of tasks.

Set USE_REWARD = True for the LC-inspired reward-modulated condition.
Set USE_REWARD = False for the baseline (no reward gating) condition.
Both conditions use the same trajectory seed so paths are identical.

"""

import os

import numpy as np

from _tem_eval import run_eval
from neuralplayground.agents.whittington_2020 import Whittington2020
from neuralplayground.agents.whittington_2020_extras import (
    whittington_2020_parameters as parameters,
)
from neuralplayground.arenas import BatchEnvironment, DiscreteObjectEnvironment
from neuralplayground.backend import SingleSim, tem_training_loop
from neuralplayground.experiments import Sargolini2006Data

# ── Experiment flag ────────────────────────────────────────────────────────────
USE_REWARD = True          # Set True for reward-modulated condition
TRAJECTORY_SEED = 42        # Fixed seed ensures identical trajectories across conditions
N_PRETRAIN_EPISODES = 50    # Episodes of free exploration before reward gating starts
REWARD_LOCATION = [3.0, 3.0]  # Reward site; inside all environment bounds
TD_ALPHA = 0.1              # TD learning rate
TD_GAMMA = 0.9              # TD discount factor
# ──────────────────────────────────────────────────────────────────────────────

_condition = "reward_modulated" if USE_REWARD else "baseline"
simulation_id = f"TEM_{_condition}_sim"
save_path = os.path.join(os.getcwd(), "results_sim", _condition)
agent_class = Whittington2020
env_class = BatchEnvironment
training_loop = tem_training_loop

params = parameters.parameters()
full_agent_params = params.copy()

arena_x_limits = [
    [-5, 5],
    [-4, 4],
    [-5, 5],
    [-6, 6],
    [-4, 4],
    [-5, 5],
    [-6, 6],
    [-5, 5],
    [-4, 4],
    [-5, 5],
    [-6, 6],
    [-5, 5],
    [-4, 4],
    [-5, 5],
    [-6, 6],
    [-5, 5],
]
arena_y_limits = [
    [-5, 5],
    [-4, 4],
    [-5, 5],
    [-6, 6],
    [-4, 4],
    [-5, 5],
    [-6, 6],
    [-5, 5],
    [-4, 4],
    [-5, 5],
    [-6, 6],
    [-5, 5],
    [-4, 4],
    [-5, 5],
    [-6, 6],
    [-5, 5],
]

room_widths = [x[1] - x[0] for x in arena_x_limits]
room_depths = [y[1] - y[0] for y in arena_y_limits]

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
    "state_densities": [discrete_env_params["state_density"]]
    * env_params["batch_size"],
    "use_behavioural_data": False,
    # Reward / TD parameters
    "use_reward": USE_REWARD,
    "reward_location": REWARD_LOCATION,
    "td_alpha": TD_ALPHA,
    "td_gamma": TD_GAMMA,
    "n_pretrain_episodes": N_PRETRAIN_EPISODES,
}

training_loop_params = {
    "n_episode": 10000,
    "params": full_agent_params,
    "trajectory_seed": TRAJECTORY_SEED,
    "random_start": False,
    "eval_fn": run_eval,
    "eval_interval": 1000,
    "eval_save_path": save_path,
}

sim = SingleSim(
    simulation_id=simulation_id,
    agent_class=agent_class,
    agent_params=agent_params,
    env_class=env_class,
    env_params=env_params,
    training_loop=training_loop,
    training_loop_params=training_loop_params,
)

# print(sim)
print("Running sim...")
sim.run_sim(save_path)
print("Sim finished.")
