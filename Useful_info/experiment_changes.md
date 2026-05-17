# Experiment Changes: LC-Inspired Reward-Modulated TEM

## Overview

Two conditions are compared: baseline TEM (no reward) and a reward-modulated TEM where
Hebbian memory updates are gated by the TD error signal (δ), mimicking locus coeruleus
neuromodulation. Both conditions follow identical trajectories (same seed), start at [0,0],
and are evaluated every 1000 episodes with plots and raw data saved for post-hoc analysis.

---

## Files Changed

### 1. `neuralplayground/agents/whittington_2020_extras/whittington_2020_model.py`

**Purpose**: Intercept the Hebbian update and scale it by ReLU(δ). No other model logic touched.

| Function | Change |
|---|---|
| `forward()` | Unpacks optional 4th element (`td_scale`) from each walk-step tuple; backward-compatible — 3-element steps still work |
| `iteration()` | Accepts `td_scale=None`; passes it through to both `hebbian()` calls |
| `hebbian()` | Accepts `td_scale=None`; when provided multiplies `eta` by `td_scale.view(-1,1,1)` before the weight update |

**Key equation change** (`hebbian`):
```
# Before
M = clamp(λ·M_prev + η·M_new)

# After (when td_scale provided)
M = clamp(λ·M_prev + η·ReLU(δ)·M_new)
```

`td_scale` is shaped `(batch_size,)` externally and reshaped to `(batch_size, 1, 1)` inside
`hebbian()` to broadcast correctly over `M_new` of shape `(batch_size, n_p, n_p)`.

---

### 2. `neuralplayground/agents/whittington_2020.py`

**New `__init__` parameters** (all optional, default to baseline-compatible values):

| Parameter | Default | Description |
|---|---|---|
| `use_reward` | `False` | Enable TD-gated Hebbian update |
| `reward_location` | `[3.0, 3.0]` | (x, y) coordinates of reward site |
| `td_alpha` | `0.1` | TD learning rate for V(s) |
| `td_gamma` | `0.9` | Discount factor |
| `n_pretrain_episodes` | `0` | Episodes of free (unmodulated) exploration before gating activates. Allows TEM to form stable structural representations first. |

**New method `_compute_reward_state_ids()`**:
Replicates the environment's `pos_to_state` logic to find the grid state index nearest to
`reward_location` for each of the 16 environments at init time. Result stored in
`self.reward_state_ids` — a list of 16 integers.

**New method `_compute_and_update_td(prev_obs, curr_obs)`**:
- Called in `batch_act()` on every accepted step when `use_reward=True`
- Reward delivered on **arrival** at reward state: `r = 1 if s' == reward_state_id else 0`
- Updates V table: `V(s) ← V(s) + α(r + γV(s') − V(s))`
- Returns `ReLU(δ)` as a float32 array of shape `(batch_size,)` — the Hebbian gate
- Appends one dict per environment to `self.step_log` for offline analysis

**Modified `reset()`**:
- Initialises `self.V` — list of `batch_size` zero arrays (one value per state per env)
- Initialises `self.td_errors = []` and `self.step_log = []`

**Modified `batch_act()`**:
- On every accepted step with `use_reward=True`: calls `_compute_and_update_td()` and
  appends the ReLU(δ) array to `self.td_errors`

**Modified `update()`**:
- `gating_active = use_reward and episode_count >= n_pretrain_episodes`
- When active: appends `td_scale` tensor as 4th element in each `model_input` step
- Increments `self.episode_count` after each backprop update
- Trims `self.td_errors` by the rollout window after each update to prevent unbounded growth

---

### 3. `neuralplayground/backend/training_loops.py`

**Modified `tem_training_loop()` signature** — new keyword arguments:

| Argument | Default | Description |
|---|---|---|
| `trajectory_seed` | `None` | Seeds `random` and `np.random` before `env.reset()` so object layouts and action sequences are identical across conditions |
| `random_start` | `False` | If False, agents always start at `[0, 0]`; if True, random start position |
| `eval_fn` | `None` | Callable `eval_fn(agent, env, episode, eval_save_path)` called every `eval_interval` episodes |
| `eval_interval` | `1000` | How often (in episodes) to call `eval_fn` |
| `eval_save_path` | `None` | Root directory passed to `eval_fn` for saving outputs |

**Note**: Both `random` and `np.random` are seeded because `generate_objects()` uses
Python's `random.randint`, while action selection uses `np.random.choice`. Seeding only
one would leave the other non-deterministic across conditions.

---

### 4. `examples/agent_examples/whittington_2020_run.py`

**Top-level flags** (only these need changing between runs):
```python
USE_REWARD = False          # True = reward-modulated, False = baseline
TRAJECTORY_SEED = 42        # Keep identical in both runs
N_PRETRAIN_EPISODES = 50    # Free exploration episodes before reward gating
REWARD_LOCATION = [3.0, 3.0]
TD_ALPHA = 0.1
TD_GAMMA = 0.9
```

- `save_path` automatically resolves to `results_sim/baseline/` or `results_sim/reward_modulated/`
- `n_episode = 10000`
- `eval_fn = run_eval` (imported from `_tem_eval.py`), `eval_interval = 1000`

---

### 5. `examples/agent_examples/_tem_eval.py` *(new file)*

Periodic evaluation module called every `eval_interval` episodes from the training loop.
Operates on **environment 0 only**. Uses the last `EVAL_STEPS = 600` steps from
`obs_history` to compute rate maps (600 steps gives good state coverage for a 10×10 grid).

**Key steps inside `run_eval(agent, env, episode, eval_save_path)`**:

1. Filter dummy initial observations (state_id == −1) from `obs_history`
2. Build single-env model input for env 0 only
3. Run forward pass with `torch.no_grad()` and `agent.tem.eval()`; restore `batch_size`
   in `agent.tem.hyper` afterwards (single-env eval temporarily overwrites it to 1)
4. Accumulate `p_inf` and `g_inf` per visited state across all forward steps
5. Average the **second half** of each state's visits (mirrors the original `analyse.rate_map`
   convention — first-half visits are noisier while the model is still settling)
6. Save outputs to `<eval_save_path>/plots/episode_<N>/`:

| File | Description |
|---|---|
| `p_rates.npy` | Place cell rate maps concatenated across frequencies, shape `(n_states, total_p_cells)`. Used by `tem_predictive_analysis.py`. |
| `v_table.npy` | V(s) array for env 0, shape `(n_states,)`. Reward condition only. |
| `trajectory.png` | Last 600 steps of env 0 trajectory with start (green), end (red), reward (gold star) |
| `value_map.png` | V(s) reshaped to 2D grid. Reward condition only. |
| `place_cells_<freq>.png` | Up to 30 place cell rate maps per frequency module |
| `grid_cells_<freq>.png` | Up to 30 grid cell rate maps per frequency module |

**`_save_rate_maps(rates, n_cells_list, room_w, room_d, ep_dir, prefix, episode)`**:
Helper that saves one PNG per frequency module. Normalises `axs` to 2D shape for uniform
indexing regardless of whether there is one row or multiple rows of subplots.

---

### 6. `examples/agent_examples/tem_predictive_analysis.py` *(new file)*

Post-hoc analysis script. Run after both training runs are complete.

```bash
cd examples/agent_examples
python tem_predictive_analysis.py
```

Outputs saved to `results_sim/predictive_analysis/`. Requires `p_rates.npy` and (for reward
condition) `v_table.npy` to be present in each episode folder — these are written by
`_tem_eval.py`.

**Three analyses**:

| Output file | What it shows |
|---|---|
| `population_activity_baseline.png` / `population_activity_reward_modulated.png` | Mean firing across all place cells per state plotted on the 2D grid at each checkpoint. Reward condition activity focus should spread backward from reward over training. |
| `value_correlation.png` | Pearson correlation between mean place cell activity per state and V(s) over training (reward condition only). A rising trend indicates cells become more predictive of future reward. |
| `peak_distance_from_reward.png` | Mean/median distance of each cell's peak-firing state from the reward location, for both conditions across episodes. Backward shift shows up as reward-condition cells developing peaks further from reward compared to baseline. |
| `peak_distance_hist_<condition>.png` | Histogram of peak-firing distances at first vs. last checkpoint for each condition. Shows whether the distribution shifts over training. |

**Grid coordinates**: The analysis reconstructs the (x, y) centre of every state for env 0
using `np.linspace` with the same formula as `DiscreteObjectEnvironment`, so state indices
map correctly to Euclidean positions when computing distances to the reward.

---

## What is NOT changed

- TEM architecture, loss function, and backpropagation
- Grid cell (g) dynamics and transition model
- Sensory encoding/decoding pathway
- Inference and generative model structure
- `BatchEnvironment` and `DiscreteObjectEnvironment`

---

## How to run the experiment

**Step 1 — Baseline**:
```python
# whittington_2020_run.py
USE_REWARD = False
```
```bash
cd examples/agent_examples
python whittington_2020_run.py
```

**Step 2 — Reward-modulated**:
```python
USE_REWARD = True
```
```bash
python whittington_2020_run.py
```

**Step 3 — Post-hoc analysis**:
```bash
python tem_predictive_analysis.py
```

Keep `TRAJECTORY_SEED = 42` identical in both runs.

---

## Output folder structure

```
results_sim/
├── baseline/
│   ├── agent, agent_hyper, arena, training_hist.dict   ← saved at end of run
│   └── plots/
│       ├── episode_1000/
│       │   ├── p_rates.npy          ← place cell rate maps (n_states, total_p_cells)
│       │   ├── trajectory.png
│       │   ├── place_cells_Theta.png ... place_cells_High_Gamma.png
│       │   └── grid_cells_Theta.png  ... grid_cells_High_Gamma.png
│       └── episode_2000/ ... episode_10000/
│
├── reward_modulated/
│   └── plots/
│       └── episode_N/
│           ├── p_rates.npy
│           ├── v_table.npy          ← V(s) for env 0  (reward condition only)
│           ├── trajectory.png
│           ├── value_map.png        ← only present in reward condition
│           ├── place_cells_*.png
│           └── grid_cells_*.png
│
└── predictive_analysis/             ← written by tem_predictive_analysis.py
    ├── population_activity_baseline.png
    ├── population_activity_reward_modulated.png
    ├── value_correlation.png
    ├── peak_distance_from_reward.png
    ├── peak_distance_hist_baseline.png
    └── peak_distance_hist_reward_modulated.png
```

---

## Per-step log fields (`agent.step_log`)

Populated only when `use_reward=True`. Each dict entry:

| Field | Description |
|---|---|
| `episode` | Episode index at time of step (incremented each `update()` call) |
| `env` | Environment index (0–15) |
| `s` | State ID at start of transition |
| `s_prime` | State ID after transition |
| `reward` | 1.0 if s' is the reward state, else 0.0 |
| `delta` | Raw TD error: `r + γV(s') − V(s)` |
| `hebbian_scale` | `max(0, delta)` — actual multiplier applied to η in that step |
| `V_s` | Value of V(s) **after** the TD update |
