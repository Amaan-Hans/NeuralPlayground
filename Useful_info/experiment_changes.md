# Experiment Changes: TEM-R — Held-Landmark TD Value via Place-Cell Bias

## Overview

Two conditions are compared: **baseline TEM** (no value mechanism) and
**TEM-R** (value-biased TEM). Both share the *same* environment layout —
including the landmark scheme described below — and the same trajectory seed,
so the environment and the path walked are identical either way; only the
agent's use of a value mechanism differs. Each condition runs for **5000
episodes**, evaluated every 1000 episodes with plots and raw data saved for
post-hoc analysis.

> This document supersedes four earlier designs: a neural value head that
> gated the Hebbian update; a tabular value head keyed by object identity;
> a tabular value keyed by state but concatenated onto the observation `x`;
> and a tabular value keyed by state, biasing place cells via `f_v` but with
> one table entry per physical grid state. All four have been fully removed —
> see "Superseded designs" at the bottom.

### Baseline TEM (no value mechanism)
- Hebbian update (uniform, unmodulated): `M = λM + η(p − p̂)(p + p̂)ᵀ`
- Same landmark-equipped environment as TEM-R (see Change 0) — landmarks
  exist in the layout either way, baseline just never attaches a value to them.

### TEM-R (current design)

**Change 0 — Landmarks: a small set of objects that never duplicate**

The root cause of every earlier indexing headache was that there are only 45
sensory objects spread across up to 144 states per environment, so most
objects necessarily repeat and a sensory observation alone can't disambiguate
which physical state — or which "zone" relative to the reward — the agent is
actually in. Design 0 fixes this at the *source*: `DiscreteObjectEnvironment`
now reserves object ids `[0, N_LANDMARKS)` (`N_LANDMARKS=10`) and places each
one at exactly one state per environment — never duplicated — sampled without
replacement, weighted toward states closer to `reward_location`:
```python
weight(state) = exp(-distance(state, reward_location) / landmark_bias_scale)
```
(`landmark_bias_scale=2.0` by default — smaller = stronger pull toward the
reward; see `DiscreteObjectEnvironment.generate_objects()` /
`_weighted_sample_without_replacement()` in
[discritized_objects.py](../neuralplayground/arenas/discritized_objects.py)).
The remaining `n_objects - N_LANDMARKS` ids are distributed over all other
states exactly as before (uniform random, with replacement — duplicates
expected and fine). Sampling uses the stdlib `random` module exclusively (not
numpy) so this stays reproducible under the same `random.seed` already used
for object generation — verified empirically: baseline and TEM-R produce
byte-for-byte identical object layouts and trajectories under the same seed.

This is enabled for **both conditions** (it's an environment property, not a
TEM-R-only feature) — see `Files Changed` §6 for why.

**Change 1 — Tabular TD value head, indexed by held landmark identity**

A lookup table `V[landmark_id]`, one `(N_LANDMARKS,)` array per environment.
Because each landmark is unique within its environment, object identity and
physical-state identity are the same thing for these 10 ids — no ambiguity,
unlike the other ~35 objects.

The agent tracks, per environment, the **most recently encountered landmark
id** (`self.held_landmark`) and keeps using it — completely unchanged — across
every non-landmark step until the next landmark is reached:
```
δ = r + γ · V[held_curr] − V[held_prev]
V[held_prev] += α · δ
```
`r = 1.0` if the new state is the nearest grid state to `reward_location`,
else `0.0`. Note this means the agent can be credited for reward *while
holding a landmark's context*, even many steps after physically leaving that
landmark's state — value reflects the total reward experienced "in that
landmark's zone," not literally at that one tile.

> **Hold-vs-decay (recorded for future reference, not yet implemented):** the
> hold is currently a hard, permanent carry-forward — no time-based decay
> toward zero the longer it's been since the landmark was last seen. A decaying
> hold (e.g. exponential fade toward 0 as steps-since-last-landmark grows) was
> considered and explicitly deferred — revisit this if the hard hold turns out
> to over-credit landmarks for reward received long after leaving their zone.

No neural network, no gradient, no separate optimiser for the table itself —
this is `neuralplayground/agents/td_value_head.py::TDValueHead`, unchanged
from earlier designs except for what `n_keys_per_env`/`keys` represent
(landmark id, sized `N_LANDMARKS`, instead of physical state, sized
`n_states`).

**Change 2 — V(landmark) biases place-cell inference, not the observation**

Unchanged from the previous design — still **not** appended to `x` (see
"Superseded designs" #3 below for why that doesn't work). `V` is passed as a
separate scalar alongside `x` and enters through an additive bias on the
inferred place-cell code:
```python
# Model.inf_p(), per frequency module f:
mu_p = self.f_p(g_[f] * x_[f])
mu_p = mu_p + self.f_v[f](v)     # only when use_value_bias=True
```
`v` is `V[held_landmark]` max-normalised to `[0,1]` per environment. This part
of the mechanism (the `f_v` bias itself) is unchanged from the previous
design — only *what* `v` represents (held landmark value vs. raw per-state
value) and *how often the underlying table changes* (only at landmark
transitions vs. every step) changed in this round.

**What is explicitly NOT done:** the Hebbian update is left completely
unmodulated in both conditions; `x`/`n_x`/`n_x_c` are identical between
conditions; the hold has no decay (see callout above).

---

## Files Changed

### 1. `neuralplayground/arenas/discritized_objects.py`

**New** (`DiscreteObjectEnvironment`): `__init__` reads `n_landmarks=0`,
`reward_location=None`, `landmark_bias_scale=2.0` from `env_kwargs` (all
optional, default reproduces the original fully-random layout exactly).
`generate_objects()` branches on `n_landmarks > 0` to reserve and place
landmark ids via the new static method `_weighted_sample_without_replacement`
(stdlib `random` only, for seed reproducibility).

### 2. `neuralplayground/agents/whittington_2020.py`

**New `__init__` parameter** `n_landmarks` (default 10) — must match the
environment's `n_landmarks`.

**New `reset()` state**: `self.held_landmark` (list, length batch_size, most
recent landmark id per env or `None`), `self.held_landmark_history` (append-only,
parallel to `obs_history` — records the context held *at* each historical
step, before that step's transition updates it).

**`td`**: `TDValueHead(n_envs=batch_size, n_keys_per_env=[n_landmarks]*batch_size, ...)`
— table size dropped from up to 144 (per-state) to `n_landmarks=10`.

**`batch_act()`**: in the `all_allowed` branch, appends the pre-update held
context to `held_landmark_history`, then updates `self.held_landmark[i]` only
when the newly reached object's id is `< n_landmarks`, then runs
`self.td.update(self.held_landmark.copy(), rewards)` — i.e. the TD key is now
held-landmark id, not raw state id.

**Re-added `_object_id(obs_entry)`** (removed in the previous round, brought
back solely to detect "is the current object a landmark").

**`_value_for_history(held_landmark_history)`**: signature changed from
`(history)` to `(held_landmark_history)` — looks up `V[key]` directly per
recorded held context instead of deriving a key from each step's raw
observation.

**`update()` / `collect_final_trajectory()`**: slice
`self.held_landmark_history[-n_rollout:]` (or `-n_walk:`) alongside `history`
and pass that slice to `_value_for_history`.

### 3. `neuralplayground/agents/td_value_head.py`

No code changes — already generic over what `n_keys_per_env`/`keys` mean.

### 4. `neuralplayground/agents/whittington_2020_extras/whittington_2020_model.py` / `whittington_2020_parameters.py`

No changes this round — the `f_v` bias mechanism from the previous design is
reused as-is.

### 5. `examples/agent_examples/_tem_eval.py`

**Forward pass**: `held_indices`/`real_indices` tracked alongside the existing
dummy-row filtering so `agent.held_landmark_history` stays aligned with the
filtered `obs_history` slice; `v_seq` now looks up `V[held_landmark]` instead
of `V[state_id]`.

**`v_table.npy`**: now built by projecting `agent.td.V[0]` (shape
`(n_landmarks,)`) onto each landmark's unique state via
`env.environments[0].objects`; all non-landmark states are `NaN` (they have no
fixed value of their own — whatever context they hold is path-dependent), not
`0`.

**`value_map.png` / `object_value_map.png`**: re-themed around landmarks —
gray background (`cmap.set_bad`) for non-landmark (`NaN`) states so the
`n_landmarks` coloured cells stand out; lime boxes mark all landmark states
(not "states sharing the reward's object," which doesn't make sense once
value is landmark-keyed); reward location marked with a cyan star on both
plots for direct visual comparison against landmark brightness.

### 6. `examples/agent_examples/whittington_2020_run.py` / `whittington_2020_loop_run.py`

**New top-level flags**: `N_LANDMARKS=10`, `LANDMARK_BIAS_SCALE=2.0`.

**`discrete_env_params`** now always includes `n_landmarks`, `reward_location`,
`landmark_bias_scale` — **unconditionally, not gated by `USE_REWARD`**. This is
a deliberate choice: landmarks are a property of the *environment*, and both
conditions must share the same environment for the baseline-vs-TEM-R
comparison to isolate the value mechanism's contribution rather than
conflating it with "landmarks are better spatial anchors than duplicated
objects" (a real, separate effect — see the caveat below).
`agent_params["n_landmarks"]` is also always passed (the agent only acts on it
when `use_reward=True`).

> **Methodological note carried over from the design discussion:** a
> never-duplicated object is inherently a better localisation anchor than a
> repeating one, *independent of whether it carries value*. Because both
> conditions now share the landmark layout, this effect is held constant
> between them — baseline vs. TEM-R isolates the value contribution. To
> separately measure the landmark-layout contribution itself, compare *this*
> baseline against the pre-landmark baseline archived in
> `results_interpretation.md` (different environment entirely, not a clean
> A/B — treat as a rough historical reference only).

### 7. `examples/agent_examples/run_full_experiment.py`

Unchanged by this round.

---

## What is NOT changed

- TEM's sensory pathway: `f_c`, `two_hot_table`, `x_prev2x`, `x2x_`, `W_tile`,
  `W_repeat`, the generative cross-entropy loss over `x`.
- The Hebbian update (always unmodulated, in both conditions).
- Grid cell (g) dynamics and transition model.
- `BatchEnvironment`.
- `n_x` / `n_x_c` (identical between conditions; never bumped).
- `Model.inf_p`'s `f_v` bias mechanism itself (reused from the previous design).

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
│   ├── td_value_table               ← pickled list of per-env V arrays (agent.td.V), shape (n_landmarks,)
│   ├── whittington_2020_model.py
│   └── plots/
│       └── episode_N/
│           ├── p_rates.npy
│           ├── v_table.npy          ← V(landmark) projected onto states, shape (n_states,), NaN elsewhere
│           ├── trajectory.png
│           ├── value_map.png        ← landmark V on 2D grid, reward marked with a star
│           ├── object_value_map.png ← + object id overlay, lime boxes = landmark states
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
  exploration before gating activated.

The results in `results_interpretation.md` were produced under this design and
do not describe the current architecture's behaviour.

### 2. Object-keyed tabular value (second design)

Table indexed by **object id** instead of physical state. Two states sharing
the same (necessarily-repeating, since there were no landmarks yet) object
were *forced* to share the same `V` — the opposite of what's needed to tell
repeated-object states apart.

### 3. State-keyed value concatenated onto the observation (third design)

Fixed design 2's indexing problem (switched to state-keyed `V`), but appended
`v_norm` as a 46th dimension on `x`, requiring `pars["n_x"]` to be widened by 1
before constructing the agent. Investigation found this injection point is
nearly inert: `Model.f_c`'s argmax-based two-hot lookup and the cross-entropy
loss's argmax-based labelling both discard the appended channel (the real
one-hot entry always wins argmax ties against a bounded `v_norm`), so `V` had
no real gradient path into the model. Motivated moving to the `f_v`
place-cell bias.

### 4. State-keyed value via the f_v place-cell bias (fourth design)

Kept the `f_v` bias mechanism (still current), but the table was indexed by
**physical state** (one entry per state, up to 144 per env) rather than by
held landmark identity, and updated using the raw current state id every
step (no "holding" — value was looked up fresh at whatever state the agent
was literally standing on). Worked, but every one of the ~35 non-landmark
objects' states still had *no* way to disambiguate themselves from their
duplicates other than the table being keyed by state rather than object —
i.e. it solved the *indexing* problem but not the underlying *sensory
ambiguity* (the agent's observation still couldn't tell two same-object states
apart; only the hand-fed `v` could). Motivated introducing actual
never-duplicated landmark objects (Change 0) so disambiguation happens at the
sensory level too, with value keyed by the resulting unique landmark identity.
