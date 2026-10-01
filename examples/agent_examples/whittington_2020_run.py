"""Training simulation for the Whittington et al. 2020 agent (TEM),
consolidated with everything learned from the size/rotation audit:

- ROTATE_ENVIRONMENTS (default True): reproduces torch_tem's own training
  regime — each batch slot's environment gets replaced with a freshly
  randomized instance (same size/config) after a variable-length walk,
  instead of staying fixed for the whole run. This is what let a matched
  25-state/5000-episode run recover ~100% one-step predictive accuracy
  (x_gt), vs ~50-78% without it. See tem_training_loop's docstring for the
  mechanism. CAVEAT: env 0's plots/checkpoint-trend analyses
  (tem_predictive_analysis.py) assume env 0 is the SAME environment across
  checkpoints — that assumption breaks under rotation, since env 0 may have
  rotated to a fresh layout between any two checkpoints. Set
  ROTATE_ENVIRONMENTS=0 if you need that assumption to hold.
- VARY_ARENA_SIZE (default True): reproduces NeuralPlayground's original
  8x8/10x10/12x12-repeating mix across the 16 batch slots (average ~103.5
  states/env) instead of one uniform size - this is the "normal"/historical
  environment-size diversity, independent of ROTATE_ENVIRONMENTS's object-
  layout diversity. Set to "0" (or set ARENA_SIDE) for a single uniform
  square size across all slots instead - useful when you specifically want
  environment size held constant as a controlled variable (e.g. the
  size/rotation audit this session ran used ARENA_SIDE=5 or 10 uniformly,
  to isolate size as one variable at a time). The original torch_tem
  reference config used 5x5 (25 states) uniformly.
- ARENA_SIDE (default 10, i.e. 100 states/env): only used when
  VARY_ARENA_SIZE=0 - all batch slots then use this single square size.
- N_CONTROL_LANDMARKS / DECOY_OBJECT_ID: extra environment structure (see
  DiscreteObjectEnvironment.generate_objects docstring) — defaults (10, 30)
  match the config the cluster runs actually used: 10 more never-duplicated,
  reward-biased "control" objects the agent's value mechanism never tracks
  (ids >= N_LANDMARKS, outside held_landmark/TD's own range), and every
  remaining state tiled with one fixed decoy id so ONLY the 20 landmark
  states are ever sensorily distinguishable. Set N_CONTROL_LANDMARKS=0 /
  DECOY_OBJECT_ID=None (env vars, empty string for the latter) to reproduce
  the original fully-random decoy layout exactly.
- SAVE_MULTIENV_CSV (default True): also saves a per-(env, state) place-cell
  CSV covering every environment in the batch (not just env 0) at episodes
  1000/2000/3000/4000/5000, via _tem_eval_allenvs.run_eval_combined - a
  chunked forward pass (verified numerically identical to the monolithic
  version) so it's safe even when both conditions checkpoint concurrently
  on one shared GPU. Set to "0" to fall back to the original env-0-only
  run_eval and skip this entirely.

Both conditions use the same landmark objects (placed with bias toward
REWARD_LOCATION) and the same trajectory seed, so the environment and the
path walked are identical either way.

Set USE_REWARD = True for the TEM-R condition: a TD-learned value over held
landmark identity is written into the dedicated trailing dimension of the
compressed sensory code x_c by Model.inference(), right after f_c's
argmax/lookup - it then flows through the rest of TEM's sensory pathway like
any other observation dimension.
Set USE_REWARD = False for the baseline condition: identical environment and
trajectory, but no value mechanism at all (isolates the value-learning
contribution from the landmark-layout contribution — see
Useful_info/experiment_changes.md).
"""

import os

from neuralplayground.agents.whittington_2020 import Whittington2020
from neuralplayground.agents.whittington_2020_extras import (
    whittington_2020_parameters as parameters,
)
from neuralplayground.arenas import BatchEnvironment, DiscreteObjectEnvironment
from neuralplayground.backend import SingleSim, tem_training_loop
from neuralplayground.experiments import Sargolini2006Data

# ── Experiment flags ───────────────────────────────────────────────────────────
# Env-var overrides let run_full_experiment.py / submitit jobs drive every
# condition from one script without editing this file; manual edits of the
# literals below still work for one-off interactive runs.
USE_REWARD          = os.environ.get("TEM_USE_REWARD", "0") == "1"      # False = baseline, True = TEM-R
TEST_MODE           = os.environ.get("TEM_TEST_MODE", "0") == "1"      # True = 10-episode smoke test
TRAJECTORY_SEED     = int(os.environ.get("TEM_SEED", "123"))
TD_ALPHA            = 0.1
TD_GAMMA            = 0.95
N_LANDMARKS         = int(os.environ.get("TEM_N_LANDMARKS", "10"))
N_CONTROL_LANDMARKS = int(os.environ.get("TEM_N_CONTROL_LANDMARKS", "10"))
_decoy_env          = os.environ.get("TEM_DECOY_OBJECT_ID", "30")      # "" (empty) = original random decoys
DECOY_OBJECT_ID     = int(_decoy_env) if _decoy_env != "" else None
LANDMARK_BIAS_SCALE = 2.0
VARY_ARENA_SIZE     = os.environ.get("TEM_VARY_ARENA_SIZE", "1") == "1"  # historical 8/10/12-mix vs uniform
ARENA_SIDE          = float(os.environ.get("TEM_ARENA_SIDE", "10"))    # only used when VARY_ARENA_SIZE=0
_SIZE_CYCLE_ENV     = os.environ.get("TEM_SIZE_CYCLE")  # e.g. "5,6,7" or "5,6,7,10,12" - overrides both of the above
ROTATE_ENVIRONMENTS = os.environ.get("TEM_ROTATE_ENVIRONMENTS", "1") == "1"
BATCH_SIZE          = int(os.environ.get("TEM_BATCH_SIZE", "16"))
SAVE_MULTIENV_CSV   = os.environ.get("TEM_SAVE_MULTIENV_CSV", "1") == "1"  # all-envs place-cell CSV per checkpoint
N_EPISODE_OVERRIDE  = os.environ.get("TEM_N_EPISODE")
EVAL_INTERVAL_OVERRIDE = os.environ.get("TEM_EVAL_INTERVAL")
# ──────────────────────────────────────────────────────────────────────────────

if SAVE_MULTIENV_CSV:
    from _tem_eval_allenvs import run_eval_combined as run_eval
else:
    from _tem_eval import run_eval

if _SIZE_CYCLE_ENV:
    _size_cycle = [float(s) for s in _SIZE_CYCLE_ENV.split(",")]
elif VARY_ARENA_SIZE:
    # Historical NeuralPlayground mix: repeats every 4 slots, average ~103.5
    # states/env (64, 100, or 144 depending on position in the cycle).
    _size_cycle = [10, 8, 10, 12]
else:
    _size_cycle = [ARENA_SIDE]

# Reward sits at a fixed diagonal coordinate, chosen to stay inside the
# SMALLEST arena in the cycle (0.5 units of margin from the wall) - this
# reduces to the historical [3.0, 3.0] exactly whenever every size in the
# cycle is >= 8 (half-width 3.5), and shrinks automatically for mixes that
# include smaller arenas (e.g. TEM_SIZE_CYCLE="5,6,7").
_min_size = min(_size_cycle)
_reward_coord = min(3.0, _min_size / 2 - 0.5)
REWARD_LOCATION = [_reward_coord, _reward_coord]

# Overrides the default results_sim<suffix>/ root when set by
# run_full_experiment.py, e.g. to write into experiments/random/seed_<N>/.
_SAVE_ROOT = os.environ.get("TEM_SAVE_ROOT")

_condition = "reward_modulated" if USE_REWARD else "baseline"
simulation_id = f"TEM_{_condition}_sim"
_results_root = "results_sim_test" if TEST_MODE else "results_sim"
save_path = os.path.join(_SAVE_ROOT, _condition) if _SAVE_ROOT else os.path.join(os.getcwd(), _results_root, _condition)
agent_class = Whittington2020
env_class = BatchEnvironment
training_loop = tem_training_loop

if N_EPISODE_OVERRIDE is not None:
    _n_episode = int(N_EPISODE_OVERRIDE)
else:
    _n_episode = 10 if TEST_MODE else 5000
if EVAL_INTERVAL_OVERRIDE is not None:
    _eval_interval = int(EVAL_INTERVAL_OVERRIDE)
else:
    _eval_interval = 2 if TEST_MODE else 1000

params = parameters.parameters()
# The eta/lambda/lr-decay/loss-weight annealing schedule below was tuned for
# params["train_it"]=20000 backprop iterations. This run only does
# _n_episode (one iteration per episode) - rescale so eta (Hebbian memory
# write rate) still reaches its mature value by the end of THIS run instead
# of stalling early (see rescale_schedule_for_train_it's docstring).
params = parameters.rescale_schedule_for_train_it(params, _n_episode)
full_agent_params = params.copy()
# TEM-R doesn't widen n_x (raw one-hot vocabulary) at all: V(landmark) reaches
# TEM by being written into the dedicated trailing dimension of the compressed
# code x_c (n_x_c is 1 wider than the two-hot identity code across BOTH
# conditions - see whittington_2020_parameters.py - so baseline and TEM-R use
# an identical network width; only whether that dimension is ever set to a
# nonzero value differs, via agent_params["use_reward"] below).

_half_cycle = [[-s / 2, s / 2] for s in _size_cycle]
arena_x_limits = [_half_cycle[i % len(_half_cycle)] for i in range(BATCH_SIZE)]
arena_y_limits = [_half_cycle[i % len(_half_cycle)] for i in range(BATCH_SIZE)]

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
    # Landmarks are part of the shared environment, not gated by USE_REWARD —
    # both conditions get the same never-duplicated, reward-biased landmark
    # layout; only the agent's use of a value mechanism differs.
    "n_landmarks": N_LANDMARKS,
    "n_control_landmarks": N_CONTROL_LANDMARKS,
    "decoy_object_id": DECOY_OBJECT_ID,
    "reward_location": REWARD_LOCATION,
    "landmark_bias_scale": LANDMARK_BIAS_SCALE,
}

env_params = {
    "environment_name": "BatchEnvironment",
    "batch_size": BATCH_SIZE,
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
    "n_landmarks": N_LANDMARKS,
}

training_loop_params = {
    "n_episode": _n_episode,
    "params": full_agent_params,
    "trajectory_seed": TRAJECTORY_SEED,
    "random_start": False,
    "eval_fn": run_eval,
    "eval_interval": _eval_interval,
    "eval_save_path": save_path,
    "rotate_environments": ROTATE_ENVIRONMENTS,
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

if __name__ == "__main__":
    _size_desc = (
        f"cycle={_size_cycle} (n_states={sorted(set(int((w[1]-w[0])**2) for w in arena_x_limits))}) "
        f"reward_location={REWARD_LOCATION}"
    )
    print(
        f"Running sim... condition={_condition} size={_size_desc} "
        f"n_episode={_n_episode} rotate_environments={ROTATE_ENVIRONMENTS} "
        f"n_control_landmarks={N_CONTROL_LANDMARKS} decoy_object_id={DECOY_OBJECT_ID}"
    )
    sim.run_sim(save_path)
    print("Sim finished.")
