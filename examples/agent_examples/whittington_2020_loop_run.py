"""TEM training with a two-phase policy: random exploration, then a fixed
square loop whose path deliberately passes through the reward-adjacent
state, with landmark objects placed explicitly along the approach to that
reward state ("on the way to the reward") rather than merely biased toward
it.

Phase 1 (first N_PHASE1_EPISODES): random exploration across all 16 envs.
Phase 2 (remaining episodes): every env follows the same fixed square loop
(same relative displacements from the same reset position, applied to all
16 envs identically - see the historical note in run_condition() below for
why this differs from an earlier version's docstring claim that envs 1-15
would keep exploring randomly in phase 2, which the code never actually did).

Square loop (starting at LOOP_START):
    4x "down" action, 3x "right", 4x "up", 3x "left" - repeating.
    NOTE: DiscreteObjectEnvironment.step() negates the y-component for pure-
    vertical actions (dx==0) - see discritized_objects.py's step(). So the
    "down"/"up" action *labels* below are the raw action vectors, not the
    actual resulting direction of travel (a "down" step actually moves the
    agent in +y). The actual path is computed programmatically below
    (_simulate_path, replicating that same sign flip) rather than assumed
    from the labels, specifically to avoid the kind of silent direction/
    offset mixup this quirk causes if you reason about it by label alone.

The path is verified - assertion, fails loudly at import time - to actually
pass through the state nearest reward_location before any training starts.
This is the "doesn't get offset" guarantee: if REWARD_LOCATION or the loop
shape are changed later such that they no longer intersect, the script
refuses to run rather than silently training a misaligned experiment.

Landmarks are placed explicitly on the states from the loop start up to and
including the reward-adjacent state ("on the way to the reward"), via the
`landmark_positions` env_kwarg (see DiscreteObjectEnvironment.generate_
objects) - not just biased toward it as in the non-loop experiment. Any
landmark ids beyond that approach segment's length fall back to the usual
reward-distance-biased random placement.

Flags
-----
TEST_MODE : True  = 100 episodes total (quick sanity check)
            False = 5 000 episodes (2 500 random + 2 500 loop)

Env-var overrides (TEM_TEST_MODE / TEM_USE_REWARD / TEM_SEED / TEM_SAVE_ROOT)
let run_loop_experiment.py drive a single condition, seed, and output root
per process without editing this file; see that script for the parallel
baseline + reward_modulated driver. Run standalone (no env vars set) for the
legacy behaviour: both conditions sequentially in this one process, then the
post-hoc analysis.

Usage
-----
    cd examples/agent_examples
    python whittington_2020_loop_run.py             # both conditions, seed 123 (legacy)
    python run_loop_experiment.py                    # both conditions in parallel, seed 42, saved under experiments/seed_42/
"""

import os
import pickle
import random
import shutil

import numpy as np

from _tem_eval import run_eval
from neuralplayground.agents.whittington_2020 import Whittington2020
from neuralplayground.agents.whittington_2020_extras import (
    whittington_2020_parameters as parameters,
)
from neuralplayground.arenas import BatchEnvironment, DiscreteObjectEnvironment
from neuralplayground.experiments import Sargolini2006Data

# ── Flags ──────────────────────────────────────────────────────────────────────
# Env-var overrides let run_loop_experiment.py drive both conditions (and a
# chosen seed/output root) from one command without editing this file; manual
# edits of the literals below still work for one-off interactive runs.
TEST_MODE = os.environ.get("TEM_TEST_MODE", "0") == "1"   # True = 100 episodes; False = 5 000 episodes
# ──────────────────────────────────────────────────────────────────────────────

TRAJECTORY_SEED       = int(os.environ.get("TEM_SEED", "123"))
REWARD_LOCATION       = [3.0, 3.0]
TD_ALPHA              = 0.1
TD_GAMMA              = 0.95
N_LANDMARKS           = 10   # unique, never-duplicated landmark objects per env (shared by both conditions)
LANDMARK_BIAS_SCALE   = 2.0  # exponential length scale biasing (non-explicit) landmarks toward REWARD_LOCATION

N_TOTAL_EPISODES  = 100 if TEST_MODE else 5_000
N_PHASE1_EPISODES = 50  if TEST_MODE else 2_500
EVAL_INTERVAL     = 10  if TEST_MODE else 500

_suffix = "_test" if TEST_MODE else ""

# Set by run_loop_experiment.py to run just one condition in this process (so
# baseline/reward_modulated can be launched as separate parallel subprocesses)
# and to redirect output under experiments/seed_<N>/ instead of the default
# results_sim_loop<suffix>/. Unset = legacy standalone behaviour: this script
# runs both conditions itself, sequentially, then the post-hoc analysis.
_ENV_USE_REWARD = os.environ.get("TEM_USE_REWARD")
_SAVE_ROOT       = os.environ.get("TEM_SAVE_ROOT")

# ── Arena setup (identical to whittington_2020_run.py) ─────────────────────────
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

# ── Square-loop action sequence and the path it actually traces ───────────────
_DOWN  = [0, -1]
_RIGHT = [1,  0]
_UP    = [0,  1]
_LEFT  = [-1, 0]
SQUARE_LOOP = [_DOWN] * 4 + [_RIGHT] * 3 + [_UP] * 4 + [_LEFT] * 3
LOOP_START  = (-0.5, -0.5)


def _simulate_path(start, actions, step_size=1):
    """Replicate DiscreteObjectEnvironment.step()'s action-to-displacement
    convention exactly (including its y-sign-flip quirk for pure-vertical
    actions: action[0] == 0 -> the y-component is negated before being
    applied) so the computed path matches what training will actually do.
    Returns one (x, y) tuple per action, not including `start` itself.
    """
    path = []
    pos = np.array(start, dtype=float)
    for a in actions:
        a = np.array(a, dtype=float)
        a_rev = np.array([0.0, -a[1]]) if a[0] == 0 else a
        norm = np.linalg.norm(a_rev)
        if norm > 0:
            a_rev = a_rev / norm
        pos = pos + step_size * a_rev
        path.append(tuple(pos))
    return path


def _nearest_state_xy(target, room_w, room_d, state_density=1):
    """Mirror Whittington2020._compute_reward_state_ids's nearest-state
    logic at the xy level, so this matches what the agent itself computes
    for reward_state_ids.
    """
    res_w, res_d = int(state_density * room_w), int(state_density * room_d)
    x_arr = np.linspace(-room_w / 2 + 0.5 / state_density, room_w / 2 - 0.5 / state_density, res_w)
    y_arr = np.linspace(-room_d / 2 + 0.5 / state_density, room_d / 2 - 0.5 / state_density, res_d)
    xy = np.stack(np.meshgrid(x_arr, y_arr), axis=-1).reshape(-1, 2)
    diff = np.sum((xy - np.array(target, dtype=float)) ** 2, axis=-1)
    return tuple(xy[np.argmin(diff)])


_full_path = [LOOP_START] + _simulate_path(LOOP_START, SQUARE_LOOP, step_size=1)
assert np.allclose(_full_path[-1], _full_path[0]), (
    f"SQUARE_LOOP is not a closed loop: starts at {_full_path[0]}, ends at {_full_path[-1]}."
)
UNIQUE_PATH_STATES = _full_path[:-1]  # 14 distinct states; drop the duplicated return-to-start

_reward_xy = _nearest_state_xy(REWARD_LOCATION, room_w=room_widths[0], room_d=room_depths[0])
_reward_idx = next(
    (i for i, p in enumerate(UNIQUE_PATH_STATES) if np.allclose(p, _reward_xy)), None
)
assert _reward_idx is not None, (
    f"Reward-adjacent state {_reward_xy} is not on the square loop path {UNIQUE_PATH_STATES} "
    f"— the loop needs repositioning so it actually reaches the reward. This check exists "
    f"specifically to catch the loop and the reward silently drifting apart if either "
    f"REWARD_LOCATION or SQUARE_LOOP/LOOP_START are changed later without the other."
)
APPROACH_STATES    = UNIQUE_PATH_STATES[: _reward_idx + 1]  # loop start ... reward-adjacent state, inclusive
LANDMARK_POSITIONS = APPROACH_STATES[:N_LANDMARKS]           # "landmarks on the way to the reward"

print(f"Square loop path ({len(UNIQUE_PATH_STATES)} states): {UNIQUE_PATH_STATES}")
print(f"Reward-adjacent state: {_reward_xy}  (path index {_reward_idx} of {len(UNIQUE_PATH_STATES)})")
print(f"Landmark positions: {len(LANDMARK_POSITIONS)} explicit (on the approach to reward), "
      f"{N_LANDMARKS - len(LANDMARK_POSITIONS)} reward-distance-biased random")
print(f"  {LANDMARK_POSITIONS}")

# ── Environment/agent params shared by both conditions ────────────────────────
params = parameters.parameters()
full_agent_params = params.copy()

discrete_env_params = {
    "environment_name": "DiscreteObject",
    "state_density": 1,
    "n_objects": params["n_x"],
    "agent_step_size": 1,
    "use_behavioural_data": False,
    "data_path": None,
    "experiment_class": Sargolini2006Data,
    "n_landmarks": N_LANDMARKS,
    "reward_location": REWARD_LOCATION,
    "landmark_bias_scale": LANDMARK_BIAS_SCALE,
    "landmark_positions": LANDMARK_POSITIONS,
}
env_params = {
    "environment_name": "BatchEnvironment",
    "batch_size": 16,
    "arena_x_limits": arena_x_limits,
    "arena_y_limits": arena_y_limits,
    "env_class": DiscreteObjectEnvironment,
    "arg_env_params": discrete_env_params,
}


def run_condition(use_reward: bool):
    """Train one condition end-to-end and save it under results_sim_loop<suffix>/.

    Historical note on phase 2: an earlier version of this script's docstring
    claimed envs 1-15 would "continue random exploration" during phase 2
    while only env 0 traced the square. The code has never actually done
    that — every env's action gets overwritten with the same loop action
    each step (see the `for i in range(n_envs): actions[i] = loop_action`
    below). Checked during this rewrite: the square's extent (x in
    [-0.5, 2.5], y in [-0.5, 3.5]) fits inside even the smallest arena used
    here (8x8, bounds [-4, 4]), so this isn't a boundary-violation bug — it's
    a deliberate-looking choice (likely to keep all_allowed reliably true
    every step, since 15 independently-random envs would otherwise often
    have at least one hit a wall and block the whole batch's progress) that
    was just never reflected in the docstring. Left as-is here since it's
    not what this request was about; flagging it in case it should change.
    """
    condition = "reward_modulated" if use_reward else "baseline"
    root = _SAVE_ROOT if _SAVE_ROOT else os.path.join(os.getcwd(), "results_sim_loop" + _suffix)
    save_path = os.path.join(root, condition)
    os.makedirs(save_path, exist_ok=True)

    agent_params = {
        "model_name": "Whittington2020",
        "params": full_agent_params,
        "batch_size": env_params["batch_size"],
        "room_widths": room_widths,
        "room_depths": room_depths,
        "state_densities": [discrete_env_params["state_density"]] * env_params["batch_size"],
        "use_behavioural_data": False,
        "use_reward": use_reward,
        "reward_location": REWARD_LOCATION,
        "td_alpha": TD_ALPHA,
        "td_gamma": TD_GAMMA,
        "n_landmarks": N_LANDMARKS,
    }

    random.seed(TRAJECTORY_SEED)
    np.random.seed(TRAJECTORY_SEED)

    env   = BatchEnvironment(**env_params)
    agent = Whittington2020(**agent_params)

    obs, _ = env.reset(random_state=False, custom_state=[0, 0])
    n_rollout = agent.pars["n_rollout"]
    n_envs    = env_params["batch_size"]
    loop_idx  = 0

    print(f"\nCondition : {condition}")
    print(f"Episodes  : {N_TOTAL_EPISODES}  (phase 1: {N_PHASE1_EPISODES}, phase 2: {N_TOTAL_EPISODES - N_PHASE1_EPISODES})")
    print(f"Save path : {save_path}")

    for episode in range(1, N_TOTAL_EPISODES + 1):

        # ── Phase transition: reset to loop start ─────────────────────────────
        if episode == N_PHASE1_EPISODES + 1:
            obs, _ = env.reset(random_state=False, custom_state=list(LOOP_START))
            print(f">> Phase 2 start (ep {episode}): env reset to {LOOP_START}, square loop begins.")

        in_phase2 = (episode > N_PHASE1_EPISODES)

        # ── Collect one rollout ────────────────────────────────────────────────
        while agent.n_walk < n_rollout:
            env0_state_pre = obs[0][0]      # env 0 state id before this step
            actions = agent.batch_act(obs)

            if in_phase2:
                loop_action = SQUARE_LOOP[loop_idx % len(SQUARE_LOOP)]
                for i in range(n_envs):
                    actions[i] = loop_action
                    agent.prev_actions[i] = loop_action

            obs, state, reward = env.step(actions, normalize_step=True)

            # Advance loop_idx whenever env 0 physically moves to a new state,
            # not just on committed (all-envs) steps — prevents env 0 from
            # wall-pressing when other envs are stuck and retrying.
            if in_phase2 and obs[0][0] != env0_state_pre:
                loop_idx += 1

        # ── Gradient update ────────────────────────────────────────────────────
        agent.update()

        # episode == 1 is always evaluated too (in addition to the regular
        # EVAL_INTERVAL cadence) so the training-progress plots have a
        # checkpoint at the very start, not just from EVAL_INTERVAL onward.
        if episode % EVAL_INTERVAL == 0 or episode == 1:
            phase = "loop" if in_phase2 else "random"
            print(f"  ep {episode:6d}/{N_TOTAL_EPISODES}  [{phase}]", flush=True)
            run_eval(agent, env, episode, save_path)

    # ── Save artefacts (mirrors whittington_2020_run.py save pattern) ─────────
    print("Saving...")
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


if __name__ == "__main__":
    if _ENV_USE_REWARD is None:
        # Legacy standalone behaviour: run both conditions sequentially in
        # this one process, then run the post-hoc analysis directly.
        run_condition(use_reward=False)
        run_condition(use_reward=True)

        _root = _SAVE_ROOT if _SAVE_ROOT else os.path.join(os.getcwd(), "results_sim_loop" + _suffix)
        _baseline_plots = os.path.join(_root, "baseline", "plots")
        _reward_plots   = os.path.join(_root, "reward_modulated", "plots")

        print("\nBoth conditions done — running predictive analysis...")
        import tem_predictive_analysis as pa
        pa.RESULTS_ROOT       = _root
        pa.BASELINE_DIR       = _baseline_plots
        pa.REWARD_DIR         = _reward_plots
        pa.OUT_DIR            = os.path.join(pa.RESULTS_ROOT, "predictive_analysis")
        pa.LOOP_START_EPISODE = N_PHASE1_EPISODES
        os.makedirs(pa.OUT_DIR, exist_ok=True)
        pa.plot_population_activity_maps()
        pa.plot_value_correlation()
        pa.plot_peak_distance()
        pa.plot_grid_scores()
        pa.plot_proximal_cell_count()
        print(f"Analysis saved to: {pa.OUT_DIR}")
    else:
        # Driven by run_loop_experiment.py: run just this one condition as an
        # isolated process (so baseline/reward_modulated can run in parallel).
        # The driver runs the post-hoc analysis itself once both finish.
        run_condition(use_reward=(_ENV_USE_REWARD == "1"))
