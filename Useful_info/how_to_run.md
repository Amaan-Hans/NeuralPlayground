# How to Run the TEM-R Experiment

All commands assume your working directory is `examples/agent_examples/` and the conda environment is `tem_env`.

```bash
cd examples/agent_examples
conda activate tem_env
```

---

## Project layout

- `examples/agent_examples/` — training/probe/analysis scripts (below) and their `results_sim*/` output.
- `Useful_info/` — this documentation, plus `tem_step_trace/` (an interactive walkthrough of one TEM forward step).
- `experiments/` — reserved for upcoming experiment work (empty as of 2026-08-13; not yet wired into any script). Update this note once it has real contents.

---

## Scripts

### 0. `run_full_experiment.py` — Run everything in one command

Drives both training runs (baseline and TEM-R, **in parallel by default** —
they write to disjoint output directories) and the post-hoc analysis script
afterward via subprocess, using env-var overrides so neither script needs
manual editing.

```bash
python run_full_experiment.py                 # full 5000-episode runs, both conditions in parallel
python run_full_experiment.py --test           # 10-episode smoke test (~1 min total)
python run_full_experiment.py --sequential     # baseline then TEM-R, one at a time
python run_full_experiment.py --skip-analysis  # stop after both trainings finish
```

**Observed runtime (RTX 5070, both conditions in parallel):** ~1h45m total for
training + basic analysis. This does **not** include the multi-env probe
(section 3b below) — that's always a separate manual step, run after this
finishes, needed only for the `reward_zone_enrichment.png` plots.

This is the recommended entry point. The per-script descriptions below are for
running steps individually (e.g. to re-run just one condition).

---

### 1. `whittington_2020_run.py` — Training

Runs one full TEM training (5 000 episodes). Switch `USE_REWARD` at the top of
the file between the two conditions, or set the env vars `TEM_USE_REWARD=1` /
`TEM_TEST_MODE=1` (these override the literals, used by `run_full_experiment.py`).

**Top-level flags:**
```python
USE_REWARD          = False       # False = baseline, True = TEM-R (V(landmark) written into x_c's value dim)
TEST_MODE           = False       # True = 10-episode smoke test
TRAJECTORY_SEED     = 123         # keep identical in both runs — see "Seed reproducibility" caveat below
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

Both conditions use the same `TRAJECTORY_SEED`. The training loop seeds both `random`
(used by `generate_objects()` for object layouts) and `np.random` (used for action
selection) immediately before `env.reset()`. Because TEM never influences which
action is taken — the agent always follows a random policy — the trajectory (which
states are visited, in which order) is **byte-for-byte identical** across baseline
and TEM-R for a given seed. The TD value table in TEM-R learns from those same
visits, but does not change them.

> **Caveat found via a two-seed comparison (2026-08-12/13, see
> `experiment_changes.md`'s "Multi-env probe" section for full numbers):** the
> reward-zone-enrichment result is **not stable across seeds** — seed=42 showed
> a significant effect for both conditions (with reward_modulated modestly
> above baseline), seed=123 showed both conditions at chance. Since baseline
> moved almost as much as reward_modulated did, at least part of this is
> landmark-placement variability (itself seed-dependent), not purely a
> value-learning effect. One `TRAJECTORY_SEED` value currently drives *three*
> separate random streams — landmark placement, the training action sequence,
> and (separately, in `tem_probe_eval_multienv.py`) the probe's own
> evaluation walk. Don't treat a single-seed comparison as conclusive; see
> `Research_proposal.md`'s multi-seed protocol.

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
appended to the raw observation. The model's sensory pathway (`Model.f_c`'s
argmax-based two-hot lookup, and the cross-entropy loss's argmax-based
labelling) turned out to discard a continuous channel tacked onto the one-hot
`x` almost entirely — see `experiment_changes.md`'s "Superseded designs"
section 3 for the full investigation.

Instead, `V` is written directly into a **dedicated trailing dimension of the
compressed sensory code `x_c`**, immediately after `Model.f_c`'s argmax/lookup
completes:
```python
# Model.inference():
x_c = self.f_c(x)
if v is not None:
    x_c = x_c.float().clone()
    x_c[:, -1] = v.view(-1)      # dedicated 11th dim, never used for identity
x_f = self.x_prev2x(x_prev, x_c)  # flows through the rest of the pipeline normally
```
`n_x_c` is **11** in both conditions (was 10) — the two-hot identity
combinatorics still only ever run over the first 10 dimensions (padded with a
permanent `0` in the trailing one), so this can never collide with any
object's own identity code. `n_x` (the raw 45-object vocabulary) is unchanged
and identical between conditions. There is **no bias term anywhere** — the
old `f_v` mechanism (`Linear(1, n_p[f])` layers added to place-cell activity)
has been deleted, not just disabled; place cells are computed identically to
vanilla TEM in both conditions, `f_p(g_ * x_)`.

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
the injected value starts out near-zero and only becomes informative as the
table is learned — there's no separate warm-up phase to configure.

---

### What it saves (into `results_sim/<condition>/`)

| File | Description |
|---|---|
| `agent` | Trained TEM weights (PyTorch `state_dict`, pickled). **Same shape in both conditions** — no `f_v.*` keys exist anywhere (design 6 removed the bias layers entirely). |
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
forward pass writes into the same `x_c` value dimension the model was
actually trained with (see "How V(landmark) reaches TEM" above).

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

### 3b. `tem_probe_eval_multienv.py` — Multi-Environment Probe (for reward-zone enrichment)

Different purpose from `tem_probe_eval.py` above: not a recovery tool. `env 0`
alone (100 states) yields too few detected place fields for a statistically
meaningful "are fields clustered near reward" test, so this script runs a
long frozen-weight random-policy walk across **all 16 environments
simultaneously**, against the already-trained agent, and pools field counts
across all of them.

```bash
python tem_probe_eval_multienv.py
```

**No flags to change for normal use** — reads everything from
`results_sim/<condition>/params.dict`/`agent_hyper`/`agent` on disk, same as
`tem_probe_eval.py`. Config constants at the top of the file if you want to
adjust:

```python
TRAJECTORY_SEED     = 123   # seeds the probe's OWN evaluation walk — separate
                             # from the training seed (see "Seed reproducibility" caveat above)
N_EPISODES          = 5000  # rollouts; total steps = N_EPISODES * n_rollout
CHUNK_EPISODES      = 10    # rollouts processed per forward-pass chunk
WARMUP_STEPS        = 500   # dropped from the start before averaging
SAVE_EVERY_EPISODES = 500   # snapshot interval (overwrites previous snapshot)
```

**How it works:** walk and rate-map accumulation are *interleaved* —
`CHUNK_EPISODES` rollouts are walked, immediately forward-passed through the
model (carrying recurrent state across chunks), folded into a running
sum/count per state per frequency module, and then that chunk's raw
observation history is discarded before the next chunk starts. Memory stays
flat regardless of `N_EPISODES` — an earlier two-phase design (walk fully,
*then* reprocess the whole history) ran out of GPU memory at only ~1200
steps, since the Hebbian memory tensor `M` (`(16, 440, 440)` per step) was
being retained for every step of the window simultaneously.

**Output:** `results_sim/predictive_analysis/probe/probe_<condition>_rates.npz`
+ `probe_<condition>_meta.pkl` — a dedicated subfolder, deliberately *not*
`results_sim/<condition>/plots/`, since that folder is scanned by several
other analyses (population activity, peak distance, grid scores, proximal
cell count) that should only ever see real training checkpoints — a
frozen-weight probe of the *final* model has no relationship to any specific
training episode. A snapshot is written every `SAVE_EVERY_EPISODES` episodes
(overwriting the same file) for visible progress/resumability; the
post-completion save is authoritative.

Calls `tem_predictive_analysis.plot_reward_zone_enrichment()` automatically
at the end.

**Approximate runtime:** ~30 minutes per condition (both conditions run
sequentially within one invocation).

**See `experiment_changes.md`'s "Multi-env probe" section for the actual
enrichment numbers and an important two-seed replication finding** — the
result was not stable between seed=42 and seed=123.

---

### 4. `tem_predictive_analysis.py` — Post-hoc Analysis

Run after **both** training runs are complete; run `tem_probe_eval_multienv.py`
first if you want the reward-zone-enrichment plots too (otherwise those two
are silently skipped with a "no probe data found" message).

```bash
python tem_predictive_analysis.py
```

Reads `p_rates.npy`/`v_table.npy`/`g_rates.npy` from every `episode_<N>`
folder in both conditions' `plots/` (training checkpoints, 1000–5000), and
`probe_<condition>_rates.npz`/`_meta.pkl` from
`results_sim/predictive_analysis/probe/` (the multi-env probe, if it's been
run — see 3b above).

**Outputs saved to `results_sim/predictive_analysis/`:**

| File | What it shows | Source |
|---|---|---|
| `population_activity_baseline.png` | Mean place cell firing per grid state across checkpoints. | training |
| `population_activity_reward_modulated.png` | Same for TEM-R — activity should shift backward from reward over training. | training |
| `value_correlation.png` | Pearson r between mean place activity and V(s) over training. A rising trend means cells become predictive of future reward. | training |
| `peak_distance_from_reward.png` | Mean/median distance of each cell's peak-firing state from the reward location, both conditions across episodes. | training |
| `peak_distance_hist_baseline.png` | Histogram of peak-firing distances at first vs last checkpoint (baseline). | training |
| `peak_distance_hist_reward_modulated.png` | Same for TEM-R. | training |
| `grid_scores.png` | Mean grid score across all grid cells over training, both conditions. | training |
| `proximal_cell_count.png` | Count of place cells with peak firing near reward, loop-phase episodes only. | training |
| `reward_zone_enrichment.png` | Field-density enrichment ratio near reward, pooled across all 16 envs, one bar per condition, with shuffle-null whiskers. | **probe** |
| `reward_zone_field_distance_hist.png` | Pooled place-field distance-from-reward histogram, one panel per condition. | **probe** |

The first 8 files track *env 0's* representation *during* the 5000-episode
training run (x-axis = training episode). The last 2 come from the
frozen-weight, all-16-envs probe *after* training finishes and have no
training-episode axis at all — don't read them as a training-progress trend.

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

# 4. Multi-env probe (optional but needed for reward_zone_enrichment.png)
python tem_probe_eval_multienv.py   # also runs the analysis below automatically

# 5. Post-hoc analysis (redundant if step 4 was run, since it already calls this)
python tem_predictive_analysis.py
```

If `.npy` files are missing after training, insert this before step 5:
```bash
python tem_probe_eval.py
```
(This recovers a single env-0 checkpoint — different from
`tem_probe_eval_multienv.py`, see section 3b above.)

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
| `n_p` | [110, 110, 88, 66, 66] | Place cells per module (440 total — reflects `n_x_c=11`, both conditions) |
| `n_g` | [30, 30, 24, 18, 18] | Grid cells per module |
| `n_x` | 45 (both conditions — never bumped) | Raw sensory (one-hot) feature dimension fed into TEM |
| `n_x_c` | 11 (both conditions) | Compressed sensory code width — 10 for two-hot identity + 1 dedicated value dimension (see "How V(landmark) reaches TEM") |
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
| Reaches TEM via | The compressed sensory code `x_c`'s dedicated trailing dimension, written in `Model.inference()` right after `f_c`'s argmax/lookup — **not** a bias term, **not** concatenated onto the raw observation `x` (see `experiment_changes.md`) |
| Consumption | Max-normalised `V(held landmark)` written into `x_c[:, -1]`, then flows through the same temporal-filtering/grid-conjunction pipeline as any other sensory dimension — the Hebbian update itself is unmodulated |
