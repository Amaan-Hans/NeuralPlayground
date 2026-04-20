"""Train two TEM models back-to-back and save both as .pkl files.

Run 1 — TD / RL model  : reward signal appended to observations, TD(0) value
         propagation, learning rate scaled by value proximity.
Run 2 — Baseline model : original unsupervised TEM, no reward channel.

Both runs use 5 000 episodes and identical environments.
Saved under results_sim/TEM_rl/ and results_sim/TEM_baseline/.
"""

import os
import pickle

import numpy as np

from neuralplayground.agents.whittington_2020 import Whittington2020
from neuralplayground.agents.whittington_2020_extras import (
    whittington_2020_parameters as parameters,
)
from neuralplayground.arenas import BatchEnvironment, DiscreteObjectEnvironment
from neuralplayground.backend import SingleSim, tem_training_loop
from neuralplayground.experiments import Sargolini2006Data

# ── Shared environment geometry ───────────────────────────────────────────────
arena_x_limits = [
    [-5, 5], [-4, 4], [-5, 5], [-6, 6],
    [-4, 4], [-5, 5], [-6, 6], [-5, 5],
    [-4, 4], [-5, 5], [-6, 6], [-5, 5],
    [-4, 4], [-5, 5], [-6, 6], [-5, 5],
]
arena_y_limits = [
    [-5, 5], [-4, 4], [-5, 5], [-6, 6],
    [-4, 4], [-5, 5], [-6, 6], [-5, 5],
    [-4, 4], [-5, 5], [-6, 6], [-5, 5],
    [-4, 4], [-5, 5], [-6, 6], [-5, 5],
]
room_widths = [x[1] - x[0] for x in arena_x_limits]
room_depths = [y[1] - y[0] for y in arena_y_limits]

N_EPISODES = 5000
BATCH_SIZE  = 16
SAVE_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results_sim")


def build_env_params(n_x):
    discrete_env_params = {
        "environment_name": "DiscreteObject",
        "state_density": 1,
        "n_objects": n_x,
        "agent_step_size": 1,
        "use_behavioural_data": False,
        "data_path": None,
        "experiment_class": Sargolini2006Data,
    }
    return {
        "environment_name": "BatchEnvironment",
        "batch_size": BATCH_SIZE,
        "arena_x_limits": arena_x_limits,
        "arena_y_limits": arena_y_limits,
        "env_class": DiscreteObjectEnvironment,
        "arg_env_params": discrete_env_params,
    }


def build_agent_params(params):
    return {
        "model_name": "Whittington2020",
        "params": params,
        "batch_size": BATCH_SIZE,
        "room_widths": room_widths,
        "room_depths": room_depths,
        "state_densities": [params["state_density"]] * BATCH_SIZE,
        "use_behavioural_data": False,
    }


def run(label, n_td_reward):
    print(f"\n{'='*60}")
    print(f"  Starting: {label}  (n_td_reward={n_td_reward})")
    print(f"{'='*60}\n")

    params     = parameters.parameters(n_td_reward=n_td_reward)
    env_params = build_env_params(params["n_x"])
    agent_params = build_agent_params(params)

    save_path = os.path.normpath(os.path.join(SAVE_ROOT, label))
    os.makedirs(save_path, exist_ok=True)

    sim = SingleSim(
        simulation_id=label,
        agent_class=Whittington2020,
        agent_params=agent_params,
        env_class=BatchEnvironment,
        env_params=env_params,
        training_loop=tem_training_loop,
        training_loop_params={"n_episode": N_EPISODES, "params": params},
    )

    sim.run_sim(save_path)

    # Dump the full params dict as a convenience .pkl alongside the model files.
    os.makedirs(save_path, exist_ok=True)
    params_path = os.path.join(save_path, "params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved params  → {params_path}")
    print(f"Model weights → {os.path.join(save_path, 'agent')}")
    print(f"Hyper dict    → {os.path.join(save_path, 'agent_hyper')}")


if __name__ == "__main__":
    # Run 1: TD / RL model (n_td_reward=1, reward at centre of each grid)
    run("TEM_rl", n_td_reward=1)

    # Run 2: Baseline unsupervised TEM (no reward channel, original architecture)
    run("TEM_baseline", n_td_reward=0)

    print("\nBoth runs complete.")
