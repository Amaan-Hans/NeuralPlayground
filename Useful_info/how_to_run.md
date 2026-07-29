# How to Run the TEM-R Experiment

All commands assume your working directory is `examples/agent_examples/` and the conda environment is `tem_env`.

```bash
cd examples/agent_examples
conda activate tem_env
```

---

## Scripts

### 0. `run_full_experiment.py` — Run everything in one command

Drives both training runs (baseline, then TEM-R) and the post-hoc analysis
script in sequence via subprocess, using env-var overrides so neither script
needs manual editing.

```bash
python run_full_experiment.py                 # full 5000-episode runs (~3h/condition on GPU)
python run_full_experiment.py --test           # 10-episode smoke test (~1 min total)
python run_full_experiment.py --skip-analysis  # stop after both trainings finish
```

This is the recommended entry point. The per-script descriptions below are for
running steps individually (e.g. to re-run just one condition).

---

### 1. `whittington_2020_run.py` — Training

Runs one full TEM training (5 000 episodes). Switch `USE_REWARD` at the top of
the file between the two conditions, or set the env vars `TEM_USE_REWARD=1` /
`TEM_TEST_MODE=1` (these override the literals, used by `run_full_experiment.py`).

**Top-level flags:**
```python
USE_REWARD          = False       # False = baseline, True = TEM-R (V(landmark) biases p via f_v)
TEST_MODE           = False       # True = 10-episode smoke test
TRAJECTORY_SEED     = 42          # keep identical in both runs
REWARD_LOCATION     = [3.0, 3.0]
TD_ALPHA            = 0.1         # tabular value-table learning rate
TD_GAMMA            = 0.95
N_LANDMARKS         = 10          # unique, never-duplicated landmark objects (both conditions share these)
LANDMARK_BIAS_SCALE = 2.0         # exponential length scale biasing landmark placement toward REWARD_LOCATION
```

**Run order:**
```bash
# Step 1 — baseline (USE_REWARD = False)
python whittington_2020_run.py

# Step 2 — TEM-R (USE_REWARD = True)
python whittington_2020_run.py
```

**Quick smoke test (10 episodes):**
Set `TEST_MODE = True` (and any value of `USE_REWARD`), then run the script.
Eval runs every 2 episodes. Cleans up in under a minute.

---

### Seed reproducibility

Both conditions use `TRAJECTORY_SEED = 42`. The training loop seeds both `random`
(used by `generate_objects()` for object layouts) and `np.random` (used for action
selection) immediately before `env.reset()`. Because TEM never influences which
action is taken — the agent always follows a random policy — the trajectory (which
states are visited, in which order) is **byte-for-byte identical** across baseline
and TEM-R. The TD value table in TEM-R learns from those same visits, but does not
change them.

---

### Landmarks: solving sensory ambiguity at the source

Only 45 sensory objects are spread across up to 144 states per environment,
so most objects repeat and the observation alone can't tell two same-object
states apart. `DiscreteObjectEnvironment` now reserves object ids
`[0, N_LANDMARKS)` and places each at exactly one state per environment —
never duplicated — sampled without replacement, weighted toward states closer
to `reward_location` (`exp(-distance / landmark_bias_scale)`; see
`generate_objects()` in
[discritized_objects.py](../neuralplayground/arenas/discritized_objects.py)).
The remaining objects still repeat freely, exactly as before.

This is enabled for **both conditions** — `discrete_env_params["n_landmarks"]`
is set unconditionally, not gated by `USE_REWARD` — so baseline and TEM-R
share the identical environment and trajectory, and only the value mechanism
differs between them. (A never-duplicated object is inherently a better
localisation anchor regardless of value, so sharing the layout is what keeps
the baseline-vs-TEM-R comparison about the value mechanism specifically — see
`experiment_changes.md`'s methodological note.)

### How V(landmark) reaches TEM

TEM-R uses a TD-learned value keyed by **held landmark identity**, not
appended to the observation. The model's sensory pathway (`Model.f_c`'s
argmax-based two-hot lookup, and the cross-entropy loss's argmax-based
labelling) turned out to discard a continuous channel tacked onto the one-hot
`x` almost entirely — see `experiment_changes.md`'s "Superseded designs"
section 3 for the full investigation. Instead, `V` is passed as a separate
scalar that biases the inferred place-cell code directly, via `f_v` layers in
`Model.inf_p()` (one `Linear(1, n_p[f])` per frequency module), created only
when `use_reward=True`. `n_x` is identical between conditions — there is no
width bump anywhere.

The table is indexed by **landmark id** (size `N_LANDMARKS=10`), not physical
state. The agent tracks the most recently encountered landmark
(`agent.held_landmark`) and **keeps using it, unchanged, with no decay**,
across every non-landmark step until the next landmark is reached — so reward
received anywhere "in a landmark's zone" credits that landmark, not just the
literal tile it occupies. *(A decaying hold — fading the held value toward
zero the longer it's been since the last landmark — was considered and
deliberately deferred; revisit if the hard hold over-credits landmarks for
reward received long after leaving their zone.)*

There is no pretrain/gating delay in this design: TD updates run from episode
0. Early in training `V` is near zero everywhere (initialised to zero), so
the `f_v` bias starts out near-zero and only becomes informative as the table
is learned — there's no separate warm-up phase to configure.

---

### What it saves (into `results_sim/<condition>/`)

| File | Description |
|---|---|
| `agent` | Trained TEM weights (PyTorch `state_dict`, pickled). **TEM-R only:** also includes `f_v.0..4.{weight,bias}` (absent in baseline). |
| `agent_hyper` | TEM hyperparameter dict (pickled) |
| `arena` | Pickled `BatchEnvironment` |
| `params.dict` | Full training metadata (`agent_class`, `agent_params`, `env_class`, `env_params`, `training_loop_params`) |
| `training_hist.dict` | Per-episode loss history |
| `whittington_2020_model.py` | Copy of the model file at save time |
| `td_value_table` | Pickled list of per-environment `V` arrays (`agent.td.V`), shape `(n_landmarks,)` each — **TEM-R only** |
| `plots/episode_<N>/` | Eval snapshots every 1 000 episodes (see `_tem_eval.py`) |

**Approximate runtime:** ~3 hours per condition on a CUDA GPU.

---

### 2. `_tem_eval.py` — Periodic Evaluation (called automatically)

Not run directly. Called by the training loop every `eval_interval=1000` episodes.

**What it saves** per checkpoint folder `plots/episode_<N>/`:

| File | Description |
|---|---|
| `p_rates.npy` | Place cell rate maps, shape `(n_states, total_p_cells)`. Used by `tem_predictive_analysis.py`. |
| `g_rates.npy` | Grid cell rate maps, shape `(n_states, total_g_cells)`. |
| `v_table.npy` | `V(landmark)` projected onto each landmark's unique state, shape `(n_states,)`, `NaN` everywhere else. **TEM-R only.** |
| `trajectory.png` | Last 500 steps of env 0 trajectory (green = start, red = end, gold star = reward). |
| `value_map.png` | Landmark `V` on the 2D grid (gray = non-landmark state), reward marked with a cyan star. **TEM-R only.** |
| `object_value_map.png` | Same heatmap + every cell's object id overlaid as text, lime boxes around the `n_landmarks` landmark states — checks whether landmarks closer to the reward end up with higher learned value. **TEM-R only.** |
| `place_cells_<freq>.png` | Up to 30 place cell rate maps per frequency module. |
| `grid_cells_<freq>.png` | Up to 30 grid cell rate maps per frequency module. |

Uses the last `EVAL_STEPS = 500` steps from `obs_history`. Only env 0 is evaluated.

When `agent.use_reward` is set, `run_eval` also builds a `V(held landmark)`
sequence for env 0 and passes it alongside (not concatenated onto) the
observation, mirroring `agent._value_for_history` — required so the eval
forward pass exercises the same `f_v` place-cell bias the model was actually
trained with.

---

### 3. `tem_probe_eval.py` — Recover Missing Checkpoints

Use this **only** if training completed but the `.npy` files are missing. Loads the
saved weights, runs ~10 000 frozen steps, then calls `run_eval` to regenerate the
endpoint checkpoint.

```bash
python tem_probe_eval.py
```

**No flags to change.** Reads everything from `results_sim/<condition>/params.dict`
and `agent_hyper` on disk.

**Limitation:** only recovers the final checkpoint (episode 10 000 label). Intermediate
checkpoints require full retraining.

**Note:** When loading a reward-modulated run, the agent is rebuilt via
`agent_class(**agent_params)`, so the TD value table (`agent.td.V`) starts fresh at
zero rather than restoring `td_value_table` from disk. This is fine for probe eval
— rate maps depend on TEM weights, and the table re-learns from the probe walk
itself (see caveat in `results_interpretation.md`).

**Approximate runtime:** ~10 minutes per condition.

---

### 4. `tem_predictive_analysis.py` — Post-hoc Analysis

Run after **both** training runs (or probe runs) are complete.

```bash
python tem_predictive_analysis.py
```

Reads `p_rates.npy` and `v_table.npy` from every `episode_<N>` folder in both
conditions.

**Outputs saved to `results_sim/predictive_analysis/`:**

| File | What it shows |
|---|---|
| `population_activity_baseline.png` | Mean place cell firing per grid state across checkpoints. |
| `population_activity_reward_modulated.png` | Same for TEM-R — activity should shift backward from reward over training. |
| `value_correlation.png` | Pearson r between mean place activity and V(s) over training. A rising trend means cells become predictive of future reward. |
| `peak_distance_from_reward.png` | Mean/median distance of each cell's peak-firing state from the reward location, both conditions across episodes. |
| `peak_distance_hist_baseline.png` | Histogram of peak-firing distances at first vs last checkpoint (baseline). |
| `peak_distance_hist_reward_modulated.png` | Same for TEM-R. |

---

## Full Run Order (from scratch)

**Recommended — single command:**
```bash
python run_full_experiment.py
```

**Equivalent manual steps:**
```bash
# 1. Quick smoke test (optional)
#    set TEST_MODE = True, run either condition
python whittington_2020_run.py

# 2. Baseline training (USE_REWARD = False, TEST_MODE = False)
python whittington_2020_run.py

# 3. TEM-R training (USE_REWARD = True, TEST_MODE = False)
python whittington_2020_run.py

# 4. Post-hoc analysis
python tem_predictive_analysis.py
```

If `.npy` files are missing after training, insert this between steps 3 and 4:
```bash
python tem_probe_eval.py
```

---

## Environment Initialisation

### BatchEnvironment (16 parallel arenas)

| Env | x limits | y limits | Grid size |
|---|---|---|---|
| 0  | [-5, 5]  | [-5, 5]  | 10 × 10 = 100 states |
| 1  | [-4, 4]  | [-4, 4]  | 8 × 8 = 64 states |
| 2  | [-5, 5]  | [-5, 5]  | 10 × 10 = 100 states |
| 3  | [-6, 6]  | [-6, 6]  | 12 × 12 = 144 states |
| 4–15 | pattern repeats (10×10, 8×8, 10×10, 12×12) | | |

### DiscreteObjectEnvironment parameters

```python
state_density       = 1        # one grid state per unit area
n_objects           = 45       # sensory feature dimension (n_x — identical in both conditions)
agent_step_size     = 1
n_landmarks         = 10       # ids [0, 10) — unique per env, never duplicated, shared by both conditions
reward_location     = [3.0, 3.0]
landmark_bias_scale = 2.0      # exp(-distance/scale) weighting toward reward_location when placing landmarks
```

**Starting position:** `[0, 0]` in both conditions (`random_start=False`).
**Reward location:** `[3.0, 3.0]` in every one of the 16 environments — a fixed
physical coordinate, mapped per-environment to the nearest grid state
(`agent._compute_reward_state_ids`). Each environment's full object layout
(including which specific states the 10 landmarks land on) is randomised
independently per environment, biased toward that environment's own version of
this coordinate; the per-environment TD table only ever sees its own
environment's landmark-reward geometry.

### TEM Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `n_rollout` | 20 | Steps per gradient update |
| `n_f` | 5 | Frequency modules |
| `n_p` | [100, 100, 80, 60, 60] | Place cells per module (400 total) |
| `n_g` | [30, 30, 24, 18, 18] | Grid cells per module |
| `n_x` | 45 (both conditions — never bumped) | Sensory feature dimension fed into TEM |
| `eta` | 0.5 | Hebbian learning rate |
| `lambda` | 0.9999 | Hebbian memory decay |

### TD value table (TEM-R only)

| Parameter | Value |
|---|---|
| Representation | Tabular, one `(n_landmarks,)` array per environment (`agent.td.V`) |
| Indexing | Held landmark id (`agent.held_landmark`) — the most recently encountered landmark, carried forward **unchanged, no decay** across non-landmark steps |
| Update rule | TD(0): `V[held_prev] += alpha * (r + gamma * V[held_curr] - V[held_prev])` |
| Update frequency | Every accepted step (online), in `batch_act()` |
| Reward | `r = 1.0` if the new state is the nearest state to `reward_location`, else `0.0` |
| Reaches TEM via | `Model.inf_p`'s `f_v` bias on the place-cell code `p` — **not** concatenated onto the observation `x` (see `experiment_changes.md`) |
| Consumption | Max-normalised `V(s_t)` appended to the observation vector fed into TEM — the Hebbian update itself is unmodulated |
