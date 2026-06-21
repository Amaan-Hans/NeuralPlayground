"""Training simulation for the Whittington et al.

2020 agent, the Tolman-Eichenbaum Machine (TEM). The TEM is a model of
the hippocampus that learns to navigate a series of environments and
solve a series of tasks.

Set USE_REWARD = True for the TEM-R condition (TD-learned object value V(o)
appended to the observation vector fed into TEM).
Set USE_REWARD = False for the baseline (unmodified observation) condition.
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

# ── Experiment flags ───────────────────────────────────────────────────────────
# Env-var overrides (TEM_USE_REWARD / TEM_TEST_MODE) let run_full_experiment.py
# drive both conditions from one script without editing this file; manual edits
# of the literals below still work for one-off interactive runs.
USE_REWARD          = os.environ.get("TEM_USE_REWARD", "0") == "1"      # False = baseline, True = TEM-R (V(o) appended to obs)
TEST_MODE           = os.environ.get("TEM_TEST_MODE", "0") == "1"     # True = 10-episode smoke test (quick sanity check)
TRAJECTORY_SEED     = 42          # Fixed seed — keep identical across conditions
REWARD_LOCATION     = [3.0, 3.0]  # Reward site; inside all environment bounds
TD_ALPHA            = 0.1         # Tabular value-table learning rate
TD_GAMMA            = 0.95         # TD discount factor
# ──────────────────────────────────────────────────────────────────────────────

_condition = "reward_modulated" if USE_REWARD else "baseline"
simulation_id = f"TEM_{_condition}_sim"
_results_root = "results_sim_test" if TEST_MODE else "results_sim"
save_path = os.path.join(os.getcwd(), _results_root, _condition)
agent_class = Whittington2020
env_class = BatchEnvironment
training_loop = tem_training_loop

params = parameters.parameters()
full_agent_params = params.copy()
# TEM-R no longer widens n_x: V(s) reaches TEM through Model.inf_p's f_v bias
# (set via agent_params["use_reward"] below), not by appending to the
# observation. n_x/n_x_c stay at their baseline values in both conditions.

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
}

_n_episode    = 10   if TEST_MODE else 5000
_eval_interval = 2   if TEST_MODE else 1000

training_loop_params = {
    "n_episode": _n_episode,
    "params": full_agent_params,
    "trajectory_seed": TRAJECTORY_SEED,
    "random_start": False,
    "eval_fn": run_eval,
    "eval_interval": _eval_interval,
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
