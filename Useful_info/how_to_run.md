# How to Run the TEM-R Experiment

All commands assume your working directory is `examples/agent_examples/` and the conda environment is `tem_env`.

```bash
cd examples/agent_examples
conda activate tem_env
```

---

## Scripts

### 1. `whittington_2020_run.py` — Training

Runs one full TEM training (5 000 episodes). Switch `USE_REWARD` at the top of the file between the two conditions.

**Top-level flags:**
```python
USE_REWARD          = False       # False = baseline, True = TEM-R (V(x)-gated)
TEST_MODE           = False       # True = 10-episode smoke test
TRAJECTORY_SEED     = 42          # keep identical in both runs
N_PRETRAIN_EPISODES = 50          # free-exploration episodes before reward gating starts
REWARD_LOCATION     = [3.0, 3.0]
TD_ALPHA            = 0.1         # value head Adam learning rate
TD_GAMMA            = 0.9
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
and TEM-R. The value head in TEM-R learns from those same visits, but does not
change them.

---

### Reward gating schedule

For the first `N_PRETRAIN_EPISODES = 50` episodes:
- The value head is **already being trained** (TD updates run from episode 0)
- But **gating is inactive** — Hebbian update is uniform, same as baseline
- This lets TEM form stable structural representations before reward modulation begins

After episode 50, `gating_active = True` and V(xₜ) gates the Hebbian update.

---

### What it saves (into `results_sim/<condition>/`)

| File | Description |
|---|---|
| `agent` | Trained TEM weights (PyTorch `state_dict`, pickled) |
| `agent_hyper` | TEM hyperparameter dict (pickled) |
| `arena` | Pickled `BatchEnvironment` |
| `params.dict` | Full training metadata (`agent_class`, `agent_params`, `env_class`, `env_params`, `training_loop_params`) |
| `training_hist.dict` | Per-episode loss history |
| `whittington_2020_model.py` | Copy of the model file at save time |
| `value_head` | Trained value head `state_dict` — **reward condition only** |
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
| `v_table.npy` | V(x) averaged per state, shape `(n_states,)`. **Reward condition only.** Computed by running `value_head(x_aug)` on each visited state in the last 500 steps and averaging over visits. |
| `trajectory.png` | Last 500 steps of env 0 trajectory (green = start, red = end, gold star = reward). |
| `value_map.png` | V(x) per state reshaped to 2D grid. **Reward condition only.** |
| `place_cells_<freq>.png` | Up to 30 place cell rate maps per frequency module. |
| `grid_cells_<freq>.png` | Up to 30 grid cell rate maps per frequency module. |

Uses the last `EVAL_STEPS = 500` steps from `obs_history`. Only env 0 is evaluated.

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

**Note:** When loading a reward-modulated run, the value head is rebuilt with random
weights (the saved `value_head` file is not loaded automatically). This is fine for
probe eval — rate maps depend on TEM weights, not the value head.

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
| `value_correlation.png` | Pearson r between mean place activity and V(x) over training. A rising trend means cells become predictive of future reward. |
| `peak_distance_from_reward.png` | Mean/median distance of each cell's peak-firing state from the reward location, both conditions across episodes. |
| `peak_distance_hist_baseline.png` | Histogram of peak-firing distances at first vs last checkpoint (baseline). |
| `peak_distance_hist_reward_modulated.png` | Same for TEM-R. |

---

## Full Run Order (from scratch)

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
n_objects       = 45       # sensory feature dimension (= n_x in TEM params)
agent_step_size = 1
```

**Starting position:** `[0, 0]` in both conditions (`random_start=False`).  
**Reward location:** `[3.0, 3.0]` — top-right quadrant of env 0.

### TEM Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `n_rollout` | 20 | Steps per gradient update |
| `n_f` | 5 | Frequency modules |
| `n_p` | [100, 100, 80, 60, 60] | Place cells per module (400 total) |
| `n_g` | [30, 30, 24, 18, 18] | Grid cells per module |
| `n_x` | 45 | Sensory feature dimension |
| `eta` | 0.5 | Hebbian learning rate |
| `lambda` | 0.9999 | Hebbian memory decay |

### Value head (TEM-R only)

| Parameter | Value |
|---|---|
| Architecture | Linear(62→32) → ReLU → Linear(32→1) |
| Input | `concat(x_onehot [45], reward_flag [1], env_id_onehot [16])` |
| Optimiser | Adam, `lr = TD_ALPHA = 0.1` |
| Update frequency | Every accepted step (online TD) |
| Gating | `ReLU(V(xₜ))` applied to η in Hebbian update |
