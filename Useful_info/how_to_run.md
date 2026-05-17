# How to Run the LC-Reward Experiment

All commands assume your working directory is `examples/agent_examples/` and the conda environment is `tem_env`.

```bash
cd examples/agent_examples
conda activate tem_env
```

---

## Scripts

### 1. `whittington_2020_run.py` — Training

Runs one full TEM training (10 000 episodes). Switch `USE_REWARD` at the top of the file between the two conditions.

**Top-level flags to change between runs:**
```python
USE_REWARD            = False   # False = baseline, True = reward-modulated
TRAJECTORY_SEED       = 42      # keep identical in both runs
N_PRETRAIN_EPISODES   = 50      # free-exploration episodes before reward gating starts
REWARD_LOCATION       = [3.0, 3.0]
TD_ALPHA              = 0.1
TD_GAMMA              = 0.9
```

**Run order:**
```bash
# Step 1 — baseline
# set USE_REWARD = False in the file, then:
python whittington_2020_run.py

# Step 2 — reward-modulated
# set USE_REWARD = True in the file, then:
python whittington_2020_run.py
```

**What it saves** (into `results_sim/<condition>/`):
- `agent` — trained TEM weights (PyTorch `state_dict`, pickled)
- `agent_hyper` — TEM hyperparameter dict (pickled)
- `arena` — pickled `BatchEnvironment`
- `params.dict` — full training metadata (`agent_class`, `agent_params`, `env_class`, `env_params`, `training_loop_params`)
- `training_hist.dict` — per-episode loss history
- `whittington_2020_model.py` — copy of the model file at save time
- `plots/episode_<N>/` — eval snapshots every 1 000 episodes (see `_tem_eval.py`)

**Approximate runtime:** ~6 hours per condition on a CUDA GPU.

---

### 2. `_tem_eval.py` — Periodic Evaluation (called automatically)

Not run directly. Called by the training loop every `eval_interval=1000` episodes via the `eval_fn` hook.

**What it saves** per checkpoint folder `plots/episode_<N>/`:

| File | Description |
|---|---|
| `p_rates.npy` | Place cell rate maps, shape `(n_states, total_p_cells)`. Used by `tem_predictive_analysis.py`. |
| `v_table.npy` | V(s) for env 0, shape `(n_states,)`. Reward condition only. |
| `trajectory.png` | Last 600 steps of env 0 trajectory (green = start, red = end, gold star = reward). |
| `value_map.png` | V(s) reshaped to 2D grid. Reward condition only. |
| `place_cells_<freq>.png` | Up to 30 place cell rate maps per frequency module. |
| `grid_cells_<freq>.png` | Up to 30 grid cell rate maps per frequency module. |

Uses the last `EVAL_STEPS = 600` steps from `obs_history`. Only env 0 is evaluated.

---

### 3. `tem_probe_eval.py` — Recover Missing Checkpoints

Use this **only** if training completed but the `.npy` files are missing (e.g. because `_tem_eval.py` was updated after training ran). Loads the saved weights, runs ~800 frozen steps, then calls `run_eval` to regenerate the endpoint checkpoint.

```bash
python tem_probe_eval.py
```

**No flags to change.** Reads everything from `results_sim/<condition>/params.dict` and `agent_hyper` on disk.

**What it produces:** `results_sim/<condition>/plots/episode_10000/` with `p_rates.npy`, `v_table.npy`, and all PNGs.

**Limitation:** only recovers the final checkpoint (episode 10 000). Intermediate checkpoints (1 000–9 000) require full retraining.

**Approximate runtime:** ~10 minutes per condition.

---

### 4. `tem_predictive_analysis.py` — Post-hoc Analysis

Run after **both** training runs (or probe runs) are complete.

```bash
python tem_predictive_analysis.py
```

Reads `p_rates.npy` and `v_table.npy` from every `episode_<N>` folder in both conditions.

**Outputs saved to `results_sim/predictive_analysis/`:**

| File | What it shows |
|---|---|
| `population_activity_baseline.png` | Mean place cell firing per grid state across checkpoints. |
| `population_activity_reward_modulated.png` | Same for reward condition — activity should shift backward from reward over training. |
| `value_correlation.png` | Pearson r between mean place activity and V(s) over training. A rising trend means cells become predictive of future reward. |
| `peak_distance_from_reward.png` | Mean/median distance of each cell's peak-firing state from the reward location, both conditions across episodes. |
| `peak_distance_hist_baseline.png` | Histogram of peak-firing distances at first vs last checkpoint (baseline). |
| `peak_distance_hist_reward_modulated.png` | Same for reward condition. |

---

## Full Run Order (from scratch)

```bash
# 1. Baseline training
#    set USE_REWARD = False
python whittington_2020_run.py

# 2. Reward-modulated training
#    set USE_REWARD = True
python whittington_2020_run.py

# 3. Post-hoc analysis
python tem_predictive_analysis.py
```

If `.npy` files are missing after training, insert this between steps 2 and 3:
```bash
python tem_probe_eval.py
```

---

## Environment Initialisation

### BatchEnvironment (16 parallel arenas)

The agent trains across 16 `DiscreteObjectEnvironment` instances simultaneously. Each has different spatial dimensions but the same `state_density=1` and `n_objects=45`.

**Arena dimensions (x and y limits):**

| Env | x limits | y limits | Grid size |
|---|---|---|---|
| 0  | [-5, 5]  | [-5, 5]  | 10 × 10 = 100 states |
| 1  | [-4, 4]  | [-4, 4]  | 8 × 8 = 64 states |
| 2  | [-5, 5]  | [-5, 5]  | 10 × 10 = 100 states |
| 3  | [-6, 6]  | [-6, 6]  | 12 × 12 = 144 states |
| 4  | [-4, 4]  | [-4, 4]  | 8 × 8 = 64 states |
| 5  | [-5, 5]  | [-5, 5]  | 10 × 10 = 100 states |
| 6  | [-6, 6]  | [-6, 6]  | 12 × 12 = 144 states |
| 7–15 | (repeats the pattern above from env 0) | | |

Pattern repeats: 10×10, 8×8, 10×10, 12×12 — environments 0–3 repeated four times.

**Object layout identity:** both conditions use `TRAJECTORY_SEED = 42`. This seeds Python's `random` module (used by `generate_objects()`) and `np.random` (used for action selection) before `env.reset()`, so all 16 environments get the same object positions and the agent follows the same action sequence in both conditions.

### DiscreteObjectEnvironment parameters

```python
state_density  = 1        # one grid state per unit area
n_objects      = 45       # number of sensory objects (= n_x in TEM params)
agent_step_size = 1       # one grid cell per step
```

**State indexing:** states are numbered row-major (x varies fastest) using:
```python
xs = np.linspace(-W/2 + 0.5/density, W/2 - 0.5/density, n_x_states)
ys = np.linspace(-D/2 + 0.5/density, D/2 - 0.5/density, n_y_states)
state_id = y_idx * n_x_states + x_idx
```

For env 0 (10 × 10): state 0 = bottom-left `(-4.5, -4.5)`, state 99 = top-right `(4.5, 4.5)`.

**Starting position:** `[0, 0]` (centre of the grid) in both conditions. Controlled by `random_start=False, custom_state=[0, 0]` in `tem_training_loop`.

**Reward location:** `[3.0, 3.0]` — top-right quadrant of env 0. The nearest grid state to this location is computed at agent init time by `_compute_reward_state_ids()` and stored in `agent.reward_state_ids[0]`.

### TEM Hyperparameters (from `whittington_2020_parameters.py`)

| Parameter | Value | Description |
|---|---|---|
| `n_rollout` | 20 | Steps collected before each gradient update |
| `n_f` | 5 | Number of frequency modules |
| `n_p` | [100, 100, 80, 60, 60] | Place cells per frequency module |
| `n_g` | [30, 30, 24, 18, 18] | Grid cells per frequency module |
| `n_x` | 45 | Sensory feature dimension (= n_objects) |
| `eta` | 0.5 | Hebbian learning rate |
| `lambda` | 0.9999 | Hebbian memory decay |

Total place cells: 100+100+80+60+60 = **400 per environment**.

`p_rates.npy` shape: `(n_states, 400)` — e.g. `(100, 400)` for env 0.
