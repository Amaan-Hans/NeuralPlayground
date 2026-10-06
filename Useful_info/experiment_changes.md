# Experiment Changes: TEM-R — Held-Landmark TD Value via Post-Argmax Observation Injection

## Overview

Two conditions are compared: **baseline TEM** (no value mechanism) and
**TEM-R** (value-augmented TEM). Both share the *same* environment layout —
including the landmark scheme described below — and the same trajectory seed,
so the environment and the path walked are identical either way; only the
agent's use of a value mechanism differs. Each condition runs for **5000
episodes**, evaluated every 1000 episodes with plots and raw data saved for
post-hoc analysis.

> This document supersedes five earlier designs: a neural value head that
> gated the Hebbian update; a tabular value head keyed by object identity;
> a tabular value keyed by state but concatenated onto the observation `x`;
> a tabular value keyed by state, biasing place cells via `f_v` with one
> table entry per physical grid state; and a tabular value keyed by held
> landmark identity, still biasing place cells via `f_v`. All five have been
> fully removed — see "Superseded designs" at the bottom.
>
> **This is design #6 (current, as of 2026-08-11).** Value no longer biases
> place-cell activity as a separate additive term at all — the `f_v`
> mechanism itself is gone. Instead, `V(held landmark)` is written directly
> into a dedicated trailing dimension of the compressed sensory code `x_c`,
> immediately after `Model.f_c`'s argmax/two-hot lookup (hence "post-argmax
> injection") — so the value signal flows through the exact same temporal-
> filtering and grid-conjunction machinery as any other sensory dimension,
> rather than being injected downstream via a hand-designed bias term. See
> "Change 2" below for the mechanism, and superseded design #5 for why the
> `f_v` bias approach was replaced rather than just kept.

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

**Change 2 — V(landmark) is written into a dedicated dimension of the
compressed sensory code, right after the argmax/two-hot lookup**

`V` is **not** appended to the raw one-hot observation `x` (see "Superseded
designs" #3 for why that doesn't work) and is **no longer** injected as a
separate additive bias on place cells either (superseded design #5 — the
`f_v` mechanism is fully removed). Instead, the compressed sensory code
`x_c` — the two-hot vector `Model.f_c()` produces via `argmax(x)` + a fixed
lookup table — is widened by exactly one dimension, reserved purely as a
value channel:

```python
# Model.inference(), immediately after x_c = self.f_c(x):
if v is not None:
    x_c = x_c.float().clone()
    x_c[:, -1] = v.view(-1)          # dedicated 11th dim, never used for identity
x_f = self.x_prev2x(x_prev, x_c)     # flows through temporal filtering...
...                                   # ...and grid-conjunction (inf_p) like any other dim
```

`v` is `V[held_landmark]` max-normalised to `[0,1]` per environment — same
normalisation as before, just a different destination. Crucially, the
two-hot **identity** combinatorics (which pair of positions encodes which
object) still only ever run over the first `n_x_c_identity=10` dimensions —
the 11th is padded with a permanent `0` in the lookup table itself, so it can
never collide with or overwrite any object's own identity code. `n_x_c` is
therefore **11** in both conditions (was 10) — baseline's value dimension is
simply always `0` (baseline never supplies a `v`), so baseline and TEM-R use
an *architecturally identical* network width; only whether a nonzero `v` is
ever written in differs. `n_x` (raw one-hot vocabulary, 45 objects) and the
environment/landmark layout are completely unaffected — this lives entirely
in the compressed-code stage, not the identity-coding stage.

Two alternative mechanisms were considered and rejected before landing here:
splitting each landmark into 2 vocabulary codes (low/high value bin) was
rejected as unnecessarily complex for a coarse binary signal (would also
require reallocating the regular-object budget, 35→25, to make room); reusing
the same two bit-positions a to-be-deleted regular object's own two-hot code
would have used was rejected because two-hot codes routinely share one of
their two positions with *other* objects' codes, risking silent corruption of
some other landmark's identity.

**What is explicitly NOT done:** the Hebbian update is left completely
unmodulated in both conditions; `n_x` is identical between conditions; the
hold has no decay (see callout above); there is no bias term anywhere in the
model — `f_v` and `use_value_bias` have been deleted, not just disabled.

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

**Design 6 changes (current):**

- `whittington_2020_parameters.py`: `n_x_c` raised from 10 to 11. The
  `two_hot_table` generation loop still runs its combinatorics over a fixed
  identity width of 10 (a local `n_x_c_identity`, not `params["n_x_c"]`), then
  pads every row with one trailing `0` — guaranteeing the 11th dimension is
  never assigned to any object's identity code. `params["use_value_bias"]`
  deleted (nothing left to gate).
- `whittington_2020_model.py`: `init_trainable()`'s `f_v = ModuleList(...)`
  block deleted entirely — no bias layers exist anywhere in the model.
  `inf_p()` no longer takes a `v` parameter and no longer has a bias-add
  branch. `inference()` gained the actual injection, immediately after
  `x_c = self.f_c(x)`: `if v is not None: x_c = x_c.float().clone();
  x_c[:, -1] = v.view(-1)`. The `.float()` cast is load-bearing — the table is
  built from plain Python ints (`Long` dtype), so assigning a continuous `v`
  into an unconverted slice silently truncates it to an integer (caught during
  implementation by a dedicated verification script, not by the smoke test).
  `iteration()`/`forward()`'s existing `v`/`td_scale` step-tuple plumbing is
  unchanged — only where `v` is *consumed* moved.

Superseded design #5's `f_v` bias mechanism (`Linear(1, n_p[f])` per
frequency module, added to `mu_p` in `inf_p()`) is fully removed, not just
disabled.

### 5. `examples/agent_examples/_tem_eval.py`

**From design 5 (unchanged by design 6):** `held_indices`/`real_indices`
tracked alongside the existing dummy-row filtering so
`agent.held_landmark_history` stays aligned with the filtered `obs_history`
slice; `v_seq` looks up `V[held_landmark]` instead of `V[state_id]`.
`v_table.npy` built by projecting `agent.td.V[0]` onto each landmark's unique
state; `value_map.png`/`object_value_map.png` themed around landmarks.

**From design 6:** no functional changes to the per-checkpoint `run_eval()`
path — `v` is still computed and threaded through exactly as before, just
consumed differently inside the model.

**Multi-env probe support (added alongside design 6, used only by
`tem_probe_eval_multienv.py`, never by the normal per-checkpoint training
eval):** `compute_multienv_rates()` — small-window (`EVAL_STEPS=500`) rate
computation across every environment in the batch, not just env 0, used by
the thin `run_eval_multienv()` save-to-`plots/`-wrapper. `run_multienv_probe()`
— the actual long-probe driver: runs a frozen-weight random walk *and*
accumulates per-state, per-frequency running sums directly, one
`chunk_episodes`-rollout chunk at a time, discarding each chunk's raw
observation history immediately (`agent.obs_history` etc. reset to `[]`
after each chunk) so memory stays flat regardless of total episode count —
critical because an early two-phase design (walk fully, *then* reprocess the
whole history) OOM'd on a 12GB GPU at only ~1200 steps once `M` (the Hebbian
memory tensor, `(16, 440, 440)` per step) is retained for every step of a
long window. Writes periodic snapshots to a caller-supplied `save_dir` via
`_save_probe_snapshot()` (overwrites the same file each time — visible
progress + resumability, not one file per checkpoint).

### 5b. `examples/agent_examples/tem_probe_eval_multienv.py` (new script, design 6)

Drives `run_multienv_probe()` against both conditions' already-trained
agents: `N_EPISODES=5000`, `CHUNK_EPISODES=10`, `WARMUP_STEPS=500` (skip the
first 500 accumulated steps — recurrent path-integration state needs a few
steps to settle after starting from `prev_iter=None`). Saves
`probe_<condition>_rates.npz` + `probe_<condition>_meta.pkl` into
**`results_sim/predictive_analysis/probe/`** — a dedicated subfolder,
deliberately *not* `results_sim/<condition>/plots/`, because that folder is
scanned by several other analyses (population activity, peak distance, grid
scores, proximal cell count) that should only ever see real training
checkpoints; a frozen-weight probe of the *final* trained model has no
relationship to any specific training episode. Calls
`tem_predictive_analysis.plot_reward_zone_enrichment()` automatically at the
end. See "Multi-env probe" section below for the full rationale and the
two-seed replication finding.

### 5c. `examples/agent_examples/tem_predictive_analysis.py`

`plot_reward_zone_enrichment()` rewritten to read one summary rate-map per
condition from `_load_probe_data()` (which reads `PROBE_DIR =
predictive_analysis/probe/`), instead of scanning multiple
`plots/episode_N/p_rates_multienv.npz` checkpoints — produces one bar per
condition, not a sweep. `_condition_multienv_stats`/`_multienv_episode_dirs`/
`_load_multienv` (the old episode-scanning versions) removed.

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

- TEM's sensory pathway structure: `f_c`'s argmax/lookup mechanism, the
  two-hot identity combinatorics, `x_prev2x`, `x2x_`, `W_tile`, `W_repeat`,
  the generative cross-entropy loss over `x`. (`n_x_c`'s *width* did change,
  10→11 — see Change 2 — but the identity-coding logic itself didn't.)
- The Hebbian update (always unmodulated, in both conditions).
- Grid cell (g) dynamics and transition model.
- `BatchEnvironment`, landmark placement, `n_x` (45, both conditions).
- Any bias term on place cells — `Model.inf_p`'s `f_v` mechanism is deleted,
  not reused; place cells are now purely `f_p(g_ * x_)` in both conditions,
  identical to vanilla TEM.

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
│   ├── agent's tem state_dict: SAME shape as baseline's — no f_v.* keys exist
│   │     in either condition anymore (design 6 removed the bias layers)
│   ├── td_value_table               ← pickled list of per-env V arrays (agent.td.V), shape (n_landmarks,)
│   ├── whittington_2020_model.py
│   └── plots/
│       └── episode_N/                (N = 1000, 2000, 3000, 4000, 5000 only — real training checkpoints)
│           ├── p_rates.npy          (n_states, 440) — 440 = sum(n_p) = [110,110,88,66,66], n_x_c=11 in both conditions
│           ├── v_table.npy          ← V(landmark) projected onto states, shape (n_states,), NaN elsewhere
│           ├── trajectory.png
│           ├── value_map.png        ← landmark V on 2D grid, reward marked with a star
│           ├── object_value_map.png ← + object id overlay, lime boxes = landmark states
│           ├── place_cells_*.png
│           └── grid_cells_*.png
│
└── predictive_analysis/
    ├── population_activity_*.png, value_correlation.png, peak_distance_*.png,
    │   grid_scores.png, proximal_cell_count.png   ← all from TRAINING checkpoints above
    ├── reward_zone_enrichment.png, reward_zone_field_distance_hist.png  ← from probe/ below, NOT training
    └── probe/                        ← written by tem_probe_eval_multienv.py, NOT training
        ├── probe_baseline_rates.npz / _meta.pkl
        └── probe_reward_modulated_rates.npz / _meta.pkl
```

**Important distinction:** everything under `predictive_analysis/` except
the two `reward_zone_*` files and the `probe/` subfolder is derived from
`plots/episode_{1000..5000}/` — i.e. tracks how *env 0's* representation
changed *during* the 5000-episode training run. The `reward_zone_*` files and
`probe/` come from a completely separate, later, frozen-weight process (see
"Multi-env probe" section below) — pooled across all 16 environments, with no
training-episode axis at all.

---

## Multi-env probe: why it exists, and a two-seed replication finding

`env 0` alone (100 states) yields too few detected place fields for a
statistically meaningful "are fields clustered near reward" test. The probe
(`tem_probe_eval_multienv.py`) runs a long frozen-weight random-policy walk
across all 16 environments simultaneously against the *already-trained*
agent, pools field counts across all of them, and computes a shuffle-null
enrichment ratio (see `plot_reward_zone_enrichment()` in
`tem_predictive_analysis.py`). Current settings: `N_EPISODES=5000`,
`CHUNK_EPISODES=10` (walk and rate-accumulation are interleaved — each chunk
of 10 rollouts is walked, forward-passed, and its raw history discarded
before the next chunk starts, so memory stays flat), `WARMUP_STEPS=500`.

**Two-seed result (2026-08-12/13), same trained architecture, only
`TRAJECTORY_SEED` changed:**

| | seed=42 | seed=123 |
|---|---|---|
| baseline | ratio=1.09, **p=0.0030** | ratio=1.03, p=0.2005 (chance) |
| reward_modulated | ratio=1.13, **p<0.0001** | ratio=1.04, p=0.1700 (chance) |

The effect is **not seed-stable** — under seed=123 both conditions collapse
to chance. Notably, **baseline moved almost as much as reward_modulated
did**, which points away from "value learning specifically became less
effective this seed" and toward something shared between both conditions:
landmark placement is itself stochastic per seed (reward-distance-biased
sampling, not a fixed layout), so a "less favourable" seed for landmark
geometry can lower enrichment in *both* conditions independent of any
learning. This is consistent with the existing methodological note above
(a never-duplicated, reward-biased-placement object is inherently a better
localisation/enrichment anchor regardless of whether it carries value).

**Open methodological issue this surfaced:** one `TRAJECTORY_SEED` value
currently drives *three* separate random-number streams at once — landmark
placement (`random.seed`, inside environment generation), the training
action sequence (`np.random.seed`, during the 5000-episode training walk),
and the probe's own evaluation walk (separately seeded in
`tem_probe_eval_multienv.py`, currently set to the same value as training).
Changing "the seed" conflates all three. A recommended follow-up before
drawing conclusions from any single-seed comparison: decouple these and hold
one fixed at a time (e.g. fix the probe's evaluation walk across every
trained model, to separate "what got learned" from "which states we
happened to sample when measuring it") — and, per `Research_proposal.md`,
move toward averaging over multiple seeds rather than trusting any one.

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

Introduced the `f_v` bias mechanism, but the table was indexed by **physical
state** (one entry per state, up to 144 per env) rather than by held landmark
identity, and updated using the raw current state id every step (no
"holding" — value was looked up fresh at whatever state the agent was
literally standing on). Worked, but every one of the ~35 non-landmark
objects' states still had *no* way to disambiguate themselves from their
duplicates other than the table being keyed by state rather than object —
i.e. it solved the *indexing* problem but not the underlying *sensory
ambiguity* (the agent's observation still couldn't tell two same-object states
apart; only the hand-fed `v` could). Motivated introducing actual
never-duplicated landmark objects (Change 0) so disambiguation happens at the
sensory level too, with value keyed by the resulting unique landmark identity.

### 5. Held-landmark value via the f_v place-cell bias (fifth design — this
document's "current" design until 2026-08-11)

Kept design 4's `f_v` bias mechanism unchanged, but switched the table's key
from raw physical state to **held landmark identity** (Change 1 above,
unchanged since) — this is the design most of this document's "Change 0" and
"Change 1" sections describe, since landmarks and held-context tracking
carried forward unchanged into design 6. What changed in design 6 was
*only* Change 2: replacing the `f_v` additive bias on place cells with
writing `v` into a dedicated dimension of the compressed sensory code,
immediately after `f_c`'s argmax/lookup. Motivation for the replacement:
value entering through the same channel as any other sensory information
("baked into observation") rather than through a hand-designed bias term
disconnected from TEM's own sensory pathway — see Change 2 above for the
full mechanism and the two alternatives considered and rejected along the
way.

---

## Update 2026-10-06: cluster training moved to `submit_experiments.py`, a NaN-corruption incident + fix, wall-block retry semantics, and an interactive playground

Everything below postdates `how_to_run.md` / `PIPELINE_REBUILD_GUIDE.md` and is
not yet reflected there — both still describe the `run_full_experiment.py` /
`results_sim/` workflow. The current training entry point for real (cluster)
runs is **`examples/agent_examples/submit_experiments.py`**, driven by
`submitit` against the Wits bigbatch Slurm cluster (`--local --test` for a
no-cluster smoke test first). It submits one job per `(seed, condition)` pair,
env-var-configured (`TEM_SEED`, `TEM_USE_REWARD`, `TEM_VARY_ARENA_SIZE`,
`TEM_SIZE_CYCLE`, `TEM_ROTATE_ENVIRONMENTS`, `TEM_N_CONTROL_LANDMARKS`,
`TEM_LOAD_CHECKPOINT`, etc. — see the script's own docstring/`--help`), and
saves to `experiments/<run_tag>/seed_<seed>[_tag]/<condition>/`, not
`results_sim/`.

**"Faithful regime" runs** referenced below (`experiments/.../seed_{7,42}_faithfulregime[...]/`)
are one such campaign: mixed/varying arena sizes across the batch
(`TEM_VARY_ARENA_SIZE=1`), seeds 7 and 42, both conditions. Treat the name as
just this campaign's `--run-tag`, not a documented code concept.

### NaN-corruption incident (fixed in `7f67178`)

`_value_for_history`'s per-step value normalisation, `v_t / v_max`, guarded
only against `v_max == 0`, not against `v_max` landing on a tiny-but-nonzero
float from accumulated TD-update noise over very long runs. When that
happened, the ratio could spike to an enormous (though finite) value,
overflowing through the network in one forward pass and corrupting weights
*and Adam's moment estimates* to NaN in a single backward pass — confirmed via
bisection to occur abruptly between one 1000-episode eval checkpoint and the
next (different episode per seed), affecting **90-100% of `reward_modulated`
parameters** once triggered. Matched baseline runs (no value channel) stayed
completely clean. **Fix:** epsilon floor on `v_max` (`> 1e-6`, not just `> 0`)
plus a final clamp of the ratio to `[0, 1]`, matching the defensive clamping
already applied to every other internal signal (`g`, `p`, `M`) elsewhere in
`whittington_2020_model.py`.

**Consequence for any saved checkpoint predating this fix:** corruption is
silent (training doesn't crash, it just produces garbage from that point on)
and is *not* retroactively repaired by the code fix — a checkpoint saved
mid- or post-corruption stays corrupted forever; only a full retrain from
that point produces clean weights. Always spot-check a `reward_modulated`
checkpoint before trusting it:
```python
import pandas as pd, torch
sd = pd.read_pickle("<save_path>/agent")
bad = [k for k, v in sd.items() if isinstance(v, torch.Tensor) and (torch.isnan(v).any() or torch.isinf(v).any())]
print(f"{len(bad)}/{len(sd)} params NaN/Inf")
```
The original `seed_42_faithfulregime/reward_modulated` checkpoint failed this
check (101/166 params, many at 100% NaN fraction); the `_fixed` retrains
(`seed_{7,42}_faithfulregime_fixed/`) pass it (0/166).

### Wall-block retry semantics in `batch_act()` — matters for any custom stepping code

`Whittington2020.batch_act()` (`whittington_2020.py:274`) does not commit a
step just because `env.step()` was called. It compares the **new** batch of
locations against `self.prev_observations`: if *any* environment in the batch
claimed a nonzero action but didn't actually move (blocked by an arena wall),
the **entire batch's** pending transition is discarded — nothing is appended
to `obs_history`/`walk_actions`, no TD update runs — and every environment
gets a fresh random action to retry, individually re-checked next call.
Training and the standard eval probes never see a "claimed move that didn't
happen" transition as a result.

This was *not* replicated by this session's interactive tool (see below)
until explicitly fixed — manually driving the model by repeatedly calling the
forward pass regardless of whether the arena silently clipped a move at a
wall fed the model constant out-of-distribution phantom transitions, which
alone was enough to drop baseline's measured accuracy from ~41% to its real
~98%+ once fixed. **Any future hand-rolled stepping code (not going through
`batch_act()`) must replicate this check**, e.g. `moved = new_loc !=
prev_loc`, and skip the model update entirely when `not moved`.

### Interactive local playground (new tool, not part of the repo)

`C:\Users\hansl\Desktop\Projects\Masters_research\tem_playground\` (sibling of
this repo, not committed) — a small Flask + vanilla-JS app wrapping two frozen
checkpoints (baseline vs reward_modulated, same seed, `batch_size=1`) side by
side for interactive/manual stepping (WASD), auto-explore, live
x_p/x_g/x_gt prediction feedback, movable landmarks, place/grid-cell rate-map
tabs, and a value/reward landmark grid (blue→red by normalised `V`). Never
trains — every step runs the frozen forward pass under `torch.no_grad()`.
Key implementation points worth remembering if resuming work on it:
- Two sessions must consume the **same pre-generated action sequence** for
  "explore N steps" to be a fair side-by-side comparison — each session
  independently drawing from the shared global numpy RNG silently
  desynchronises their walks.
- Must replicate the wall-block retry check above (see `model_session.py`'s
  `_advance()`).
- `agent_params["params"]["batch_size"]` (nested) must be overridden to `1`
  separately from the top-level `agent_params["batch_size"]` — missing the
  nested one causes `IndexError` deep inside `initialise()`.

### Finding (2026-10-06): `reward_modulated` training outcome is seed-dependent, independent of arena size — seed 42 converged badly, seed 7 didn't

Proper batched probe (`tem_probe_mixed_env.py`, not the interactive tool),
both checkpoints confirmed NaN/Inf-clean:

| condition | seed | size | x_g | x_gt (last 500 eps) |
|---|---|---|---|---|
| baseline | 42 | 10 | 98.65% | **98.50%** |
| reward_modulated | 42 | 10 | 34.00% | **22.38%** |
| reward_modulated | 42 | 8 | 35.92% | **22.69%** |
| reward_modulated | 7 | 8 | — | **99.85%** (near-baseline) |

x_p (sensory) is saturated at 100% in every row, so the gap is entirely in
grid/place prediction. Seed 42's `reward_modulated` scores ~22-23% at **both**
size 8 and size 10 — ruling out "size 10 specifically is hard for this
checkpoint." This is **not** the wall-block or NaN-corruption bug (checkpoint
verified clean; interactive-tool result with the wall-block fix applied,
35-38%, sits in the same degraded regime as the proper probe, not wildly
different from it). **Conclusion: this is a seed-dependent training-outcome
difference, not a size effect or a residual implementation bug** — seed 42's
`reward_modulated` run converged to a genuinely much worse solution than seed
7's, for reasons not yet investigated (optimisation landscape / bad luck in
the value-channel interaction is the leading hypothesis, untested). Any
single-seed `reward_modulated` result should be treated as potentially
unrepresentative until more seeds are checked — echoes the existing
"not seed-stable" caveat in the two-seed multi-env-probe finding above.

**Verification note:** the obvious objection — "maybe the 22% number is
itself an artifact of the exact same NaN-clipping bug the fix addressed,
just showing up at eval/inference time instead of during training" — was
checked directly, not assumed away. `tem_probe_mixed_env.py` (the scratchpad
probe script used for the table above) turned out to independently
reimplement the value normalisation rather than calling the agent's own
`_value_for_history`, and still had the **old, unfixed** version (`v_t /
v_max if v_max > 0 else 0.0`, no epsilon floor, no `[0,1]` clamp) baked into
it. Patched it to match the real fix, then reran seed 42 and seed 7 at the
same size (8) side by side: seed 42 came back at **22.69%** — bit-for-bit
the same as before the probe-script patch — while seed 7 stayed at 99.89%.
The fix changing nothing confirms `v_max` never actually hit the dangerous
near-zero case in either walk, so the low score is not a probe-harness
artifact. Lesson: any hand-rolled probe that reimplements value-channel
normalisation instead of calling the agent's real method needs to be
checked against the current fixed version by hand — it will not pick up
future fixes automatically.

### Cluster retrain launched (2026-10-06): third seed, same conditions, run-tag `faithfulregime_v2`

To get a third data point on whether `reward_modulated`'s outcome is
seed-dependent, submitted 6 jobs (seeds 42, 7, 1 × baseline/reward_modulated)
via `submit_experiments.py`, replicating seed 7's exact saved `params.dict`
config rather than guessing: 50k episodes, `rotate_environments=True`,
`scale_walk_by_size=False` with `walk_it_min/max=100/250`, room size cycle
`[8,9,10,11]` across the 16 batch slots, `n_landmarks=10`,
`n_control_landmarks=10`, `decoy_object_id=None`, `reward_location=[3,3]`,
`td_alpha=0.1`, `td_gamma=0.95`, `eval_interval=1000`. Command:
```
python submit_experiments.py --seeds 42 7 1 --conditions baseline reward_modulated \
  --size-cycle "8,9,10,11" --no-walk-scale-by-size --walk-it-min 100 --walk-it-max 250 \
  --n-episode 50000 --eval-interval 1000 --decoy-object-id -1 \
  --run-tag faithfulregime_v2 --time-hours 48 --gres ''
```
(`--gres ''` needed — see the GRES gotcha added to `PIPELINE_REBUILD_GUIDE.md`.)
Job IDs 64660-64665, saves to
`experiments/random/seed_{42,7,1}_faithfulregime_v2/{baseline,reward_modulated}/`
on the cluster. `--run-tag` deliberately new (not `_fixed`) so this doesn't
overwrite the existing reference checkpoints. Check `squeue -u ahanslod` for
progress; expect 48h+ given 50k episodes with rotation enabled.
