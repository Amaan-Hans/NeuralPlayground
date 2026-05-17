# Experiment Changes: LC-Inspired Reward-Modulated TEM

## Overview

Two conditions are compared: baseline TEM (no reward) and a reward-modulated TEM where
Hebbian memory updates are gated by the TD error signal (δ), mimicking locus coeruleus
neuromodulation. Both conditions follow identical trajectories (same seed) and start at
position [0, 0].

---

## Files Changed

### 1. `neuralplayground/agents/whittington_2020_extras/whittington_2020_model.py`

**Purpose**: Intercept the Hebbian update and scale it by ReLU(δ).

| Function | Change |
|---|---|
| `forward()` | Unpacks optional 4th element (`td_scale`) from each walk step tuple |
| `iteration()` | Accepts `td_scale=None`; passes it to `hebbian()` |
| `hebbian()` | Accepts `td_scale=None`; multiplies `eta` by `td_scale.view(-1,1,1)` when not None |

**Key equation change** (hebbian):
```
# Before
M = clamp(λ·M_prev + η·M_new)

# After (when td_scale provided)
M = clamp(λ·M_prev + η·ReLU(δ)·M_new)
```
All other model logic (g updates, loss, sensory pathway) is **untouched**.

---

### 2. `neuralplayground/agents/whittington_2020.py`

**New `__init__` parameters** (all optional with defaults):
| Parameter | Default | Description |
|---|---|---|
| `use_reward` | `False` | Enable TD-gated Hebbian update |
| `reward_location` | `[3.0, 3.0]` | (x, y) coordinates of reward site |
| `td_alpha` | `0.1` | TD learning rate for V(s) |
| `td_gamma` | `0.9` | Discount factor |
| `n_pretrain_episodes` | `0` | Episodes of unmodulated exploration before gating activates |

**New method `_compute_reward_state_ids()`**:
Finds the grid state index nearest to `reward_location` for each of the 16 environments
at init time, replicating the environment's `pos_to_state` logic.

**New method `_compute_and_update_td(prev_obs, curr_obs)`**:
- Called inside `batch_act()` on every valid step (when `use_reward=True`)
- Delivers `r=1` when agent arrives at reward state, else `r=0`
- Updates V table: `V(s) ← V(s) + α(r + γV(s') - V(s))`
- Returns `ReLU(δ)` per environment as a float32 array
- Appends a log entry to `self.step_log` with: episode, env, s, s', reward, δ, hebbian_scale, V(s)

**Modified `reset()`**:
- Initialises `self.V` (one zero array per environment)
- Initialises `self.td_errors = []` and `self.step_log = []`

**Modified `batch_act()`**:
- When a valid step is accepted and `use_reward=True`: calls `_compute_and_update_td()`
  and appends the ReLU(δ) array to `self.td_errors`

**Modified `update()`**:
- Checks whether `episode_count >= n_pretrain_episodes` to decide if gating is active
- When active: appends `td_scale` tensor as 4th element in each `model_input` step
- Increments `self.episode_count` after each update
- Trims `self.td_errors` by removing the consumed rollout window

---

### 3. `neuralplayground/backend/training_loops.py`

**Modified `tem_training_loop()`**:
- Added `import numpy as np`
- Reads optional params: `trajectory_seed` (int) and `random_start` (bool, default False)
- If `trajectory_seed` is set: calls `np.random.seed()` before the loop so both
  conditions follow identical action sequences
- Calls `env.reset(random_state=False, custom_state=[0, 0])` by default (fixed start)

---

### 4. `examples/agent_examples/whittington_2020_run.py`

**New top-level flags** (change these to switch conditions):
```python
USE_REWARD = False          # True = reward-modulated, False = baseline
TRAJECTORY_SEED = 42        # Same value in both run files → identical trajectories
N_PRETRAIN_EPISODES = 50    # Free exploration before reward gating starts
REWARD_LOCATION = [3.0, 3.0]
TD_ALPHA = 0.1
TD_GAMMA = 0.9
```

`simulation_id` is set automatically to `"TEM_reward_sim"` or `"TEM_baseline_sim"`.

`training_loop_params` now includes `trajectory_seed` and `random_start=False`.

`agent_params` now includes all reward/TD parameters.

---

## What is NOT changed

- TEM architecture, loss function, and backpropagation
- Grid cell (g) dynamics and transition model
- Sensory encoding/decoding pathway
- Inference and generative model structure
- `BatchEnvironment` and `DiscreteObjectEnvironment` (no changes)

---

## How to run the experiment

**Baseline (no reward)**:
```python
USE_REWARD = False
```

**Reward-modulated**:
```python
USE_REWARD = True
```

Keep `TRAJECTORY_SEED = 42` the same in both runs. The comparison is always
within-environment across the 16 parallel arenas.

---

## Per-step log fields (`agent.step_log`)

Each dict entry contains:
- `episode`: episode index (incremented each `update()` call)
- `env`: environment index (0–15)
- `s`: state ID at start of transition
- `s_prime`: state ID after transition
- `reward`: 1.0 if s' is reward state, else 0.0
- `delta`: raw TD error `r + γV(s') - V(s)`
- `hebbian_scale`: `max(0, delta)` — the actual multiplier applied to η
- `V_s`: value of V(s) after the update
