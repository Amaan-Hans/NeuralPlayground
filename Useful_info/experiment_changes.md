# Experiment Changes: TEM-R — State-Keyed TD Value via Place-Cell Bias

## Overview

Two conditions are compared: **baseline TEM** (no reward) and **TEM-R** (value-
biased TEM). Both conditions follow identical trajectories (same seed), start
at `[0,0]`, and are evaluated every 1000 episodes with plots and raw data saved
for post-hoc analysis. Each condition runs for **5000 episodes**.

> This document supersedes three earlier designs: a neural value head that
> gated the Hebbian update, a tabular value head keyed by object identity, and
> a tabular value keyed by state but concatenated onto the observation `x`.
> All three have been fully removed — see "Superseded designs" at the bottom.

### Baseline TEM (no changes to original TEM)
- Observation `x`: environmental features only (45-dim object one-hot)
- Hebbian update (uniform, unmodulated): `M = λM + η(p − p̂)(p + p̂)ᵀ`

### TEM-R (current design)

**Change 1 — Tabular TD value head, indexed by physical grid state**

A lookup table `V[s]`, one array per environment sized to that environment's
actual state count (100/64/100/144 depending on room size — `agent.n_states`),
indexed by **state id** rather than object identity. Since there are only 45
objects spread across up to 144 states per environment, many states are
indistinguishable by their sensory object alone (see `how_to_run.md`'s note on
object repetition). Keying `V` by state id breaks that ambiguity: two states
holding the identical object can still carry different values depending on
their own distance from the reward.

Updated via plain TD(0) every accepted step in `batch_act()`:
```
δ = r + γ · V[s_curr] − V[s_prev]
V[s_prev] += α · δ
```
`r = 1.0` if the new state is the nearest grid state to `reward_location`
(same fixed `[x, y]` coordinate in every one of the 16 environments — see
`how_to_run.md`), else `0.0`. No neural network, no gradient, no separate
optimiser for the table itself — this is
`neuralplayground/agents/td_value_head.py::TDValueHead`. Because consecutive
states in a rollout are always spatially adjacent (the agent moves one step at
a time), this is also a literal backward propagation of value through the
grid graph, one edge at a time.

**Change 2 — V(s) biases place-cell inference, not the observation**

`V(s)` is **not** appended to `x`. Investigation showed the model's sensory
pathway is fundamentally hostile to a continuous channel riding along with the
one-hot object code:
- `Model.f_c(x)` ([whittington_2020_model.py](../neuralplayground/agents/whittington_2020_extras/whittington_2020_model.py))
  compresses `x` by `torch.argmax(x, dim=1)` then a **fixed** lookup into
  `two_hot_table` — any appended continuous value is discarded; only the
  identity of the max entry matters, and the true one-hot entry (`1.0`) always
  wins argmax ties against a bounded `v_norm ∈ [0,1]`.
- The generative cross-entropy loss derives `labels = torch.argmax(x, 1)` —
  same story, the appended channel essentially never becomes the "true label."
- The only place the raw value leaked through was a no-gradient heuristic
  error term inside `inf_g` (comparing `x` against a generated `x_hat`) — not
  a designed signal path.

Instead, `V(s)` is passed as a **separate scalar alongside** `x` (never
concatenated into it) and enters through a new additive bias on the inferred
place-cell code:
```python
# Model.inf_p(), per frequency module f:
mu_p = self.f_p(g_[f] * x_[f])
mu_p = mu_p + self.f_v[f](v)     # NEW — only when use_value_bias=True
```
`self.f_v` is a `ModuleList` of `Linear(1, n_p[f])` layers (one per frequency
module), created in `Model.init_trainable()` only when
`hyper["use_value_bias"]=True`, so the baseline model's parameter count and
checkpoint shape are completely unaffected. `v` is `V(s)` max-normalised to
`[0,1]` per environment (same normalisation rationale as before, now applied
to a real continuous input rather than a one-hot slot). This gives `V` a
genuine, gradient-carrying path into `p` — and from there into the loss,
memory, and everything downstream — without touching `n_x`, `n_x_c`, or any of
the fixed combinatorial/tiling matrices (`two_hot_table`, `W_tile`, `W_repeat`).

**What is explicitly NOT done:** the Hebbian update is left completely
unmodulated in both conditions; `x`/`n_x`/`n_x_c` are identical between
conditions. TEM-R only differs from baseline in an additive bias on `p`.

---

## Files Changed

### 1. `neuralplayground/agents/whittington_2020_extras/whittington_2020_model.py`

**This file is now changed** (previously untouched across all earlier
designs). Threaded a new optional `v` parameter through the forward chain:
`forward()` → `iteration()` → `inference()` → `inf_p()`, reusing the existing
optional-tuple-element convention (`step_data[3]` was already reserved for the
unused `td_scale`/Hebbian-gating slot from the very first superseded design;
`v` is `step_data[4]`).

- **`init_trainable()`**: creates `self.f_v` (`ModuleList` of
  `Linear(1, n_p[f])`, one per frequency) only when
  `self.hyper.get("use_value_bias", False)`.
- **`inf_p(self, x_, g_, v=None)`**: adds `self.f_v[f](v)` to `mu_p` per
  frequency when `v is not None and use_value_bias`. `x_`/`g_` computation is
  untouched.
- **`inference()` / `iteration()` / `forward()`**: pass `v` through unchanged
  otherwise.
- `hebbian()`'s `td_scale` parameter (from the earliest superseded design)
  remains present and still unused — harmless dead capability, not removed,
  not exercised.

---

### 2. `neuralplayground/agents/whittington_2020_extras/whittington_2020_parameters.py`

Added `params["use_value_bias"] = False` as the documented default (agent
overrides to `True` when `use_reward=True`).

---

### 3. `neuralplayground/agents/td_value_head.py`

Unchanged from the previous (state-keyed) design — `TDValueHead(n_envs,
n_keys_per_env, alpha=0.1, gamma=0.9)`, one table per environment sized to
that environment's state count, keyed by physical state id.

---

### 4. `neuralplayground/agents/whittington_2020.py`

**`__init__`**: sets `self.pars["use_value_bias"] = self.use_reward` before
constructing `Model` (moved `self.use_reward` assignment earlier so it's
available at that point). No more `n_x`/`obs_dim` bumping logic — `self.pars`
is identical between conditions except for this one flag.

**Removed**: `obs_dim` attribute, `_augment_observations()`.

**New method `_value_for_history(history)`**: returns a list of
`(batch_size,)` tensors (one per rollout step), each `V(s_t)` max-normalised
per environment. Does not touch observations at all.

**`update()` / `collect_final_trajectory()`**: build `v_steps =
self._value_for_history(history)` when `use_reward`, and append `[None,
v_steps[i]]` (the unused `td_scale` slot, then `v`) to each `model_input`
step — instead of modifying the observation tensor or its reshape width
(`obs_array` always reshapes to `self.pars["n_x"]` now, in both conditions).

**`batch_act()`**: unchanged — still does the state-keyed TD(0) backup
independently of how `V` reaches TEM.

---

### 5. `examples/agent_examples/whittington_2020_run.py` / `whittington_2020_loop_run.py`

**Removed**: the `full_agent_params["n_x"] = params["n_x"] + 1` bump. `n_x`
and `discrete_env_params["n_objects"]` are identical between conditions now —
TEM-R is purely a runtime flag (`use_reward=True`) passed to the agent.

---

### 6. `examples/agent_examples/_tem_eval.py`

**Forward pass**: builds a separate `v_seq` (mirroring
`agent._value_for_history`) and appends `[None, v]` to each `model_input` step
instead of concatenating `v` onto the observation array.

**Value map / object overlay** (`v_table.npy`, `value_map.png`,
`object_value_map.png`): unchanged — these still read `agent.td.V[0]`
directly, since the TD table itself is unaffected by how `V` reaches TEM.

---

### 7. `examples/agent_examples/run_full_experiment.py`

Unchanged by this round. Orchestrates both training runs (in parallel by
default — `--sequential` for one-at-a-time) then `tem_predictive_analysis.py`.
See `how_to_run.md` for usage.

---

## What is NOT changed

- TEM's sensory pathway: `f_c`, `two_hot_table`, `x_prev2x`, `x2x_`, `W_tile`,
  `W_repeat`, the generative cross-entropy loss over `x` — completely
  untouched, in both conditions.
- The Hebbian update (always unmodulated, in both conditions).
- Grid cell (g) dynamics and transition model.
- `BatchEnvironment` and `DiscreteObjectEnvironment`.
- `n_x` / `n_x_c` (identical between conditions; never bumped).

---

## How to run the experiment

See `how_to_run.md`. Quick version:
```bash
cd examples/agent_examples
conda activate tem_env
python run_full_experiment.py
```

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
│   ├── agent's tem state_dict also includes f_v.{0..4}.{weight,bias} — absent in baseline
│   ├── td_value_table               ← pickled list of per-env V arrays (agent.td.V), one entry per state
│   ├── whittington_2020_model.py
│   └── plots/
│       └── episode_N/
│           ├── p_rates.npy
│           ├── v_table.npy          ← V(s), shape (n_states,)
│           ├── trajectory.png
│           ├── value_map.png        ← V(s) plotted on 2D grid
│           ├── object_value_map.png ← V(s) + object id overlay, highlights repeated-object states
│           ├── place_cells_*.png
│           └── grid_cells_*.png
│
└── predictive_analysis/
    └── (written by tem_predictive_analysis.py)
```

---

## Superseded designs (historical record — no longer in the codebase)

### 1. Neural value head + Hebbian gating (earliest design)

- **Observation:** `x_aug = concat(x_onehot [45], reward_flag [1], env_id_onehot [16])`
  → shape `(62,)`, fed only to the value head (TEM still received plain 45-dim `x`).
- **Value function:** `value_head: Linear(62→32) → ReLU → Linear(32→1)`, trained
  via semi-gradient TD with its own Adam optimiser, `lr=td_alpha`.
- **Hebbian gating:** `M = λM + η · ReLU(V(xₜ)) · (p − p̂)(p + p̂)ᵀ` — high-value
  states encoded more durably into memory.
- **Pretrain phase:** `n_pretrain_episodes=50` episodes of unmodulated
  exploration before gating activated, to let TEM form stable structural
  representations first.

The results in `results_interpretation.md` were produced under this design and
do not describe the current architecture's behaviour.

### 2. Object-keyed tabular value (second design)

Same observation-concatenation mechanism as design 3 below, but the table was
indexed by **object id** (`argmax` of the one-hot observation) instead of
physical state. Two states sharing the same object were *forced* to share the
same `V` — the opposite of what's needed to tell repeated-object states apart.

### 3. State-keyed value concatenated onto the observation (third design)

Fixed design 2's indexing problem (switched to state-keyed `V`), but still
appended `v_norm` as a 46th dimension on `x`:
```
x_aug = concat(x_onehot [45], v_norm [1])   →   shape (46,)
```
requiring the caller to widen `pars["n_x"]` by 1 before constructing the agent.
Investigation (see Change 2 above) found this injection point is nearly inert:
`Model.f_c`'s argmax-based two-hot lookup and the cross-entropy loss's
argmax-based labelling both discard the appended channel, so `V` had no real
gradient path into the model. This motivated moving to the `f_v` place-cell
bias in the current design, which keeps the state-keyed table but changes
*where* it joins the model.
