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
USE_REWARD          = False       # False = baseline, True = TEM-R (V(s) appended to observation)
TEST_MODE           = False       # True = 10-episode smoke test
TRAJECTORY_SEED     = 42          # keep identical in both runs
REWARD_LOCATION     = [3.0, 3.0]
TD_ALPHA            = 0.1         # tabular value-table learning rate
TD_GAMMA            = 0.95
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

### How V(s) reaches TEM

TEM-R uses a TD-learned, **state-keyed** value `V(s_t)`, but it is **not**
appended to the observation. The model's sensory pathway (`Model.f_c`'s
argmax-based two-hot lookup, and the cross-entropy loss's argmax-based
labelling) turned out to discard a continuous channel tacked onto the one-hot
`x` almost entirely — see `experiment_changes.md`'s "Superseded designs"
section 3 for the full investigation. Instead, `V(s_t)` is passed as a
separate scalar that biases the inferred place-cell code directly, via new
`f_v` layers in `Model.inf_p()` (one `Linear(1, n_p[f])` per frequency
module), created only when `use_reward=True`. `n_x` is identical between
conditions — there is no width bump anywhere anymore.

The table itself is still indexed by physical grid state, not object identity
— since only 45 objects are spread across up to 144 states per environment,
many states share the same sensory object, and keying `V` by state (rather
than by object) is what lets those otherwise-identical states carry different
values, independent of how that value reaches TEM.

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
| `td_value_table` | Pickled list of per-environment `V` arrays (`agent.td.V`), one entry per state in that environment (sizes vary: 100/64/100/144) — **TEM-R only** |
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
| `v_table.npy` | `V(s)`, shape `(n_states,)`. **TEM-R only.** This is just `agent.td.V[0][:n_states]` directly — the table is already state-indexed, no projection needed. |
| `trajectory.png` | Last 500 steps of env 0 trajectory (green = start, red = end, gold star = reward). |
| `value_map.png` | `V(s)` reshaped to 2D grid. **TEM-R only.** |
| `object_value_map.png` | `V(s)` heatmap with each cell's object id overlaid as text, and lime boxes around every state sharing the reward state's object — checks whether repeated-object states actually end up with *different* values. **TEM-R only.** |
| `place_cells_<freq>.png` | Up to 30 place cell rate maps per frequency module. |
| `grid_cells_<freq>.png` | Up to 30 grid cell rate maps per frequency module. |

Uses the last `EVAL_STEPS = 500` steps from `obs_history`. Only env 0 is evaluated.

When `agent.use_reward` is set, `run_eval` also builds a `V(s)` sequence for
env 0 and passes it alongside (not concatenated onto) the observation, mirroring
`agent._value_for_history` — required so the eval forward pass exercises the
same `f_v` place-cell bias the model was actually trained with.

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
state_density   = 1        # one grid state per unit area
n_objects       = 45       # sensory feature dimension (base n_x, unaffected by TEM-R's +1 input widening)
agent_step_size = 1
```

**Starting position:** `[0, 0]` in both conditions (`random_start=False`).
**Reward location:** `[3.0, 3.0]` in every one of the 16 environments — a fixed
physical coordinate, mapped per-environment to the nearest grid state
(`agent._compute_reward_state_ids`). Because each environment's object layout is
randomised independently, the sensory *object* occupying that location differs
across environments; the per-environment TD table only ever sees its own
environment's object-reward pairing.

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
| Representation | Tabular, one array per environment sized to that env's state count (`agent.td.V`, sizes 100/64/100/144) |
| Indexing | Physical state id — **not** object identity (two states sharing an object can differ) |
| Update rule | TD(0): `V[s_prev] += alpha * (r + gamma * V[s_curr] - V[s_prev])` |
| Update frequency | Every accepted step (online), in `batch_act()` |
| Reward | `r = 1.0` if the new state is the nearest state to `reward_location`, else `0.0` |
| Reaches TEM via | `Model.inf_p`'s `f_v` bias on the place-cell code `p` — **not** concatenated onto the observation `x` (see `experiment_changes.md`) |
| Consumption | Max-normalised `V(s_t)` appended to the observation vector fed into TEM — the Hebbian update itself is unmodulated |
