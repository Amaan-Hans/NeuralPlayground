# Experiment Changes: TEM-R — Reward Observation + TD Value Gating

## Overview

Two conditions are compared: **baseline TEM** (no reward) and **TEM-R** (reward-modulated
TEM). Both conditions follow identical trajectories (same seed), start at `[0,0]`, and
are evaluated every 1000 episodes with plots and raw data saved for post-hoc analysis.
Each condition runs for **5000 episodes**.

### Baseline TEM (no changes to original TEM)
- Observation `x`: environmental features only (45-dim object one-hot)
- Hebbian update (uniform, no gating):
  `M = λM + η(p − p̂)(p + p̂)ᵀ`

### TEM-R (three changes)

**Change 1 — Extend observation x**
A reward indicator scalar and a one-hot environment identity are appended to `x` before
being fed to the value head:
```
x_aug = concat(x_onehot [45], reward_flag [1], env_id_onehot [16])   →   shape (62,)
```
`reward_flag = 1` if the current state is the reward location, `0` elsewhere.
`env_id_onehot` is a 16-dim one-hot that tells the shared value head which of the 16
parallel environments this observation came from.  Without it, the same object cue
would appear at different reward-distances in different environments, producing
conflicting TD gradients that average to a near-constant V output.
TEM itself still processes the original 45-dim `x` (architecture unchanged).

**Change 2 — Neural TD value head**
A small MLP takes `x_aug` as input and outputs a scalar `V(xₜ)`:
```
value_head: Linear(62→32) → ReLU → Linear(32→1)
```
Updated at every step via semi-gradient TD:
```
δₜ = rₜ + γ · stop_grad(V(x_{t+1})) − V(xₜ)
loss = 0.5 · δₜ²   (over batch of 16 environments)
```
Separate Adam optimiser with `lr = td_alpha`.  V is bootstrapped from the
raw observation so the head learns which sensory states predict future reward
without any explicit state-index lookup.

**Change 3 — Gate Hebbian update by V(xₜ)**
```
M = λM + η · ReLU(V(xₜ)) · (p − p̂)(p + p̂)ᵀ
```
`ReLU(V(xₜ))` ensures the gate is non-negative. High-value states (predictive
of reward) are encoded more durably into hippocampal memory.

---

## Files Changed

### 1. `neuralplayground/agents/whittington_2020_extras/whittington_2020_model.py`

No architecture changes needed. The existing `hebbian()` interface already accepts
an optional `td_scale` tensor that multiplies `η`:
```python
if td_scale is not None:
    eta = eta * td_scale.view(-1, 1, 1)
```
`td_scale` is now `ReLU(V(xₜ))` instead of `ReLU(δ)`, but the model code is
unchanged.

---

### 2. `neuralplayground/agents/whittington_2020.py`

**New `__init__` parameters** (all optional, default to baseline-compatible values):

| Parameter | Default | Description |
|---|---|---|
| `use_reward` | `False` | Enable V(x)-gated Hebbian update |
| `reward_location` | `[3.0, 3.0]` | (x, y) coordinates of reward site |
| `td_alpha` | `0.1` | Value head learning rate (Adam) |
| `td_gamma` | `0.9` | Discount factor |
| `n_pretrain_episodes` | `0` | Episodes of free (unmodulated) exploration before gating activates |

**New attributes** (when `use_reward=True`):

| Attribute | Description |
|---|---|
| `value_head` | `nn.Sequential(Linear(n_x+1+batch_size, 32), ReLU, Linear(32,1))` — input is 62-dim |
| `value_optimizer` | `Adam(value_head.parameters(), lr=td_alpha)` |

**Removed**: tabular `self.V` (list of per-env V arrays).

**New method `_build_aug_obs(obs_entry, env_idx)`**:
- Constructs `x_aug = concat(x_onehot, reward_flag, env_id_onehot)` — shape `(n_x+1+batch_size,)` = `(62,)`
- The 16-dim env one-hot disambiguates environments so the shared head learns distinct per-env value gradients
- Handles dummy initial observations (state_id=−1) by substituting zeros

**Rewritten method `_compute_and_update_td(prev_obs, curr_obs)`**:
- Batches all 16 environments into a single forward pass through `value_head`
- Performs semi-gradient TD update via `value_optimizer`
- Returns `ReLU(V(xₜ))` as a `(batch_size,)` float32 array for Hebbian gating

**Modified `reset()`**:
- Removed tabular V initialisation; `value_head` keeps weights across resets

**Modified `save_agent()`**:
- Also pickles `value_head.state_dict()` to `<save_dir>/value_head`

**`update()`** (unchanged interface):
- `gating_active = use_reward and episode_count >= n_pretrain_episodes`
- When active: appends `ReLU(V(xₜ))` tensor as 4th element in each model step
- `episode_count` incremented after each backprop update

---

### 3. `examples/agent_examples/whittington_2020_run.py`

**Top-level flags:**
```python
USE_REWARD          = False       # False = baseline, True = TEM-R
TEST_MODE           = False       # True = 10-episode smoke test
TRAJECTORY_SEED     = 42          # Keep identical across conditions
N_PRETRAIN_EPISODES = 50
REWARD_LOCATION     = [3.0, 3.0]
TD_ALPHA            = 0.1
TD_GAMMA            = 0.9
```

- `n_episode = 5000` (full run) or `10` (TEST_MODE)
- `eval_interval = 1000` (full run) or `2` (TEST_MODE)
- `save_path` resolves to `results_sim/baseline/` or `results_sim/reward_modulated/`

---

### 4. `examples/agent_examples/_tem_eval.py`

**Value map** (`v_table.npy` + `value_map.png`):
- Old: read from `agent.V[0]` (tabular V, one entry per state)
- New: run `value_head(x_aug)` for every visited state in `history_slice`, average
  over visits per state, save the resulting `(n_states,)` array

Plot label updated from `V(s)` to `V(x)` to reflect neural function approximation.

---

## What is NOT changed

- TEM architecture, loss function, and backpropagation
- Grid cell (g) dynamics and transition model
- Sensory encoding/decoding pathway (n_x=45, unchanged)
- Inference and generative model structure
- `BatchEnvironment` and `DiscreteObjectEnvironment`
- `whittington_2020_model.py` (no code changes)

---

## How to run the experiment

```bash
cd examples/agent_examples
conda activate tem_env
```

**Quick smoke test (10 episodes):**
```python
# set TEST_MODE = True, USE_REWARD = False (or True), then:
python whittington_2020_run.py
```

**Step 1 — Baseline (5000 episodes):**
```python
# set TEST_MODE = False, USE_REWARD = False
python whittington_2020_run.py
```

**Step 2 — TEM-R (5000 episodes):**
```python
# set USE_REWARD = True
python whittington_2020_run.py
```

**Step 3 — Post-hoc analysis:**
```bash
python tem_predictive_analysis.py
```

Keep `TRAJECTORY_SEED = 42` identical in both runs.

---

## Output folder structure

```
results_sim/
├── baseline/
│   ├── agent, agent_hyper, arena, params.dict, training_hist.dict
│   ├── whittington_2020_model.py
│   └── plots/
│       ├── episode_1000/
│       │   ├── p_rates.npy          (n_states, total_p_cells)
│       │   ├── g_rates.npy
│       │   ├── trajectory.png
│       │   ├── place_cells_*.png
│       │   └── grid_cells_*.png
│       └── episode_2000/ ... episode_5000/
│
├── reward_modulated/
│   ├── agent, agent_hyper, arena, params.dict, training_hist.dict
│   ├── value_head                   ← saved value_head state_dict
│   ├── whittington_2020_model.py
│   └── plots/
│       └── episode_N/
│           ├── p_rates.npy
│           ├── v_table.npy          ← V(x) averaged per state, shape (n_states,)
│           ├── trajectory.png
│           ├── value_map.png        ← V(x) plotted on 2D grid
│           ├── place_cells_*.png
│           └── grid_cells_*.png
│
└── predictive_analysis/
    └── (written by tem_predictive_analysis.py)
```

---

## Per-step log fields (`agent.step_log`)

Populated only when `use_reward=True`. Each dict entry:

| Field | Description |
|---|---|
| `episode` | Episode index at time of step |
| `env` | Environment index (0–15) |
| `s` | State ID at start of transition |
| `s_prime` | State ID after transition |
| `reward` | 1.0 if s' is the reward state, else 0.0 |
| `V_t` | Raw `V(x_t)` from value head (may be negative early in training) |
| `V_t1` | Raw `V(x_{t+1})` (bootstrapped target, no gradient) |
| `hebbian_scale` | `max(0, V_t)` — actual multiplier applied to η |
