"""TEM training with a two-phase policy.

Phase 1 (first N_PHASE1_EPISODES):  random exploration across all 16 envs.
Phase 2 (remaining episodes):        env 0 follows a fixed square loop;
                                      envs 1-15 continue random exploration.

Square loop (applied to env 0 only):
    4× down, 3× right, 4× up, 3× left — repeating from (-0.5, -0.5).
The grid resets to (-0.5, -0.5) at the phase transition.

Flags
-----
USE_REWARD   : False = baseline TEM, True = reward-modulated TEM (LC-inspired)
TEST_MODE    : True  = 100 episodes total (quick sanity check)
               False = 10 000 episodes

Usage
-----
    cd examples/agent_examples
    python whittington_2020_loop_run.py
"""

import os
import random

import numpy as np

from _tem_eval import run_eval
from neuralplayground.agents.whittington_2020 import Whittington2020
from neuralplayground.agents.whittington_2020_extras import (
    whittington_2020_parameters as parameters,
)
from neuralplayground.arenas import BatchEnvironment, DiscreteObjectEnvironment
from neuralplayground.backend import SingleSim
from neuralplayground.experiments import Sargolini2006Data

# ── Flags ──────────────────────────────────────────────────────────────────────
USE_REWARD  = True  # False = baseline, True = reward-modulated
TEST_MODE   = True    # True = 100 episodes; False = 10 000 episodes
# ──────────────────────────────────────────────────────────────────────────────

TRAJECTORY_SEED       = 42
N_PRETRAIN_EPISODES   = 50
REWARD_LOCATION       = [3.0, 3.0]
TD_ALPHA              = 0.1
TD_GAMMA              = 0.9

N_TOTAL_EPISODES  = 100   if TEST_MODE else 10_000
N_PHASE1_EPISODES = 50    if TEST_MODE else 5_000
EVAL_INTERVAL     = 10    if TEST_MODE else 1_000

_condition = "reward_modulated" if USE_REWARD else "baseline"
_suffix    = "_test" if TEST_MODE else ""
save_path  = os.path.join(os.getcwd(), "results_sim_loop" + _suffix, _condition)
os.makedirs(save_path, exist_ok=True)

# ── Square-loop action sequence (env 0 only, phase 2) ─────────────────────────
_DOWN  = [0, -1]
_RIGHT = [1,  0]
_UP    = [0,  1]
_LEFT  = [-1, 0]
SQUARE_LOOP = [_DOWN] * 4 + [_RIGHT] * 3 + [_UP] * 4 + [_LEFT] * 3

# ── Environment & agent setup (identical to whittington_2020_run.py) ──────────
params = parameters.parameters()
full_agent_params = params.copy()

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
    "use_reward": USE_REWARD,
    "reward_location": REWARD_LOCATION,
    "td_alpha": TD_ALPHA,
    "td_gamma": TD_GAMMA,
    "n_pretrain_episodes": N_PRETRAIN_EPISODES,
}

# ── Build env and agent directly (no SingleSim — we need a custom loop) ───────
random.seed(TRAJECTORY_SEED)
np.random.seed(TRAJECTORY_SEED)

env   = BatchEnvironment(**env_params)
agent = Whittington2020(**agent_params)

obs, _ = env.reset(random_state=False, custom_state=[0, 0])
n_rollout = agent.pars["n_rollout"]
n_envs    = env_params["batch_size"]
loop_idx  = 0

print(f"Condition : {_condition}")
print(f"Episodes  : {N_TOTAL_EPISODES}  (phase 1: {N_PHASE1_EPISODES}, phase 2: {N_TOTAL_EPISODES - N_PHASE1_EPISODES})")
print(f"Save path : {save_path}")
print()

for episode in range(1, N_TOTAL_EPISODES + 1):

    # ── Phase transition: reset env 0 to loop start ───────────────────────────
    if episode == N_PHASE1_EPISODES + 1:
        obs, _ = env.reset(random_state=False, custom_state=[-0.5, -0.5])
        print(f">> Phase 2 start (ep {episode}): env reset to (-0.5, -0.5), square loop begins.")

    in_phase2 = (episode > N_PHASE1_EPISODES)

    # ── Collect one rollout ───────────────────────────────────────────────────
    while agent.n_walk < n_rollout:
        n_walk_before  = agent.n_walk   # capture before batch_act — n_walk increments inside it
        env0_state_pre = obs[0][0]      # env 0 state id before this step

        actions = agent.batch_act(obs)

        # All envs follow the loop in phase 2 so all_allowed fires on almost
        # every step (all moving the same direction = no random wall collisions).
        # Env 0 only: loop_idx advances when env 0 physically moves.
        if in_phase2:
            loop_action = SQUARE_LOOP[loop_idx % len(SQUARE_LOOP)]
            for i in range(n_envs):
                actions[i] = loop_action
                agent.prev_actions[i] = loop_action

        obs, state, reward = env.step(actions, normalize_step=True)

        # Advance loop_idx whenever env 0 physically moves to a new state,
        # not just on committed (all-envs) steps.  This prevents env 0 from
        # wall-pressing when other envs are stuck and retrying.
        if in_phase2 and obs[0][0] != env0_state_pre:
            loop_idx += 1

    # ── Gradient update ───────────────────────────────────────────────────────
    agent.update()

    if episode % EVAL_INTERVAL == 0:
        phase = "loop" if in_phase2 else "random"
        print(f"  ep {episode:6d}/{N_TOTAL_EPISODES}  [{phase}]", flush=True)
        run_eval(agent, env, episode, save_path)

# ── Save artefacts (mirrors whittington_2020_run.py save pattern) ─────────────
import pickle, shutil

print("\nSaving...")

with open(os.path.join(save_path, "agent"), "wb") as f:
    pickle.dump(agent.tem.state_dict(), f)

with open(os.path.join(save_path, "agent_hyper"), "wb") as f:
    pickle.dump(agent.tem.hyper, f)

training_dict = {
    "agent_class":  Whittington2020,
    "agent_params": agent_params,
    "env_class":    BatchEnvironment,
    "env_params":   env_params,
}
with open(os.path.join(save_path, "params.dict"), "wb") as f:
    pickle.dump(training_dict, f)

src_model = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "neuralplayground", "agents", "whittington_2020_extras",
    "whittington_2020_model.py",
)
if os.path.exists(src_model):
    shutil.copy(src_model, os.path.join(save_path, "whittington_2020_model.py"))

print(f"Done. Results in: {save_path}")
