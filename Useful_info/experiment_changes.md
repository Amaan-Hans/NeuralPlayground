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
overwrite the existing reference checkpoints.

### Second NaN-corruption incident (2026-10-06, same campaign) — not reward-specific after all

Two of these six jobs (seed 42 and seed 1, both `reward_modulated`) hit the
exact same NaN-corruption signature again: clean for ~15-26k episodes, then
an abrupt 0%->90-100% jump in NaN parameters between one 1000-episode
checkpoint and the next. Checked directly whether this was a recurrence of
the `7f67178` value-ratio bug: it wasn't — `v_table.npy` (the actual TD
value table) was verified perfectly clean at the exact episode where the
model weights themselves had already gone ~98% NaN, ruling that mechanism
out. Root cause instead: **`whittington_2020.py`'s training loop had no
gradient clipping at all** between `loss.backward()` and `self.adam.step()`
— a single rare large-loss chunk can produce one unclamped Adam update
large enough to blow shared weights (`w_x`, `alpha`, `g_init`, `w_p`, ...)
straight to NaN/Inf. Fixed with `clip_grad_norm_(max_norm=10.0)` plus a
skip-the-optimiser-step guard (logs `[NaN-GUARD]`) if the grad norm itself
is already non-finite — commit `f125ac4`.

**Initially assumed this was reward_modulated-specific** (matching the
original incident's pattern) and only relaunched those two jobs with the
fix, leaving the other four running on unfixed code. **That assumption was
wrong**: a follow-up check caught **seed 1's `baseline`** hitting the
identical signature (ep 14000->15000) on the still-unfixed job. Baseline has
no value channel at all — its `x_c`'s trailing dimension is always exactly
0 — so this definitively rules out anything value-channel-related as the
cause. It's a general unclipped-gradient risk in the core architecture that
can hit **any** condition; `reward_modulated` showing it first across two
incidents now looks like small-sample coincidence, not a real asymmetry.

All 6 jobs were killed and relaunched from episode 0 under the fix, same
run-tag `faithfulregime_v2_gradclip`: 64884 (seed42 RM), 64885 (seed1 RM),
64908 (seed1 baseline), 64909 (seed42 baseline), 64911 (seed7 baseline),
64912 (seed7 RM). Check `squeue -u ahanslod` for progress, and grep each
job's `run.log` for `[NaN-GUARD]` if it recurs even with clipping — if so,
`max_norm=10.0` may need lowering, or gradient magnitude alone isn't the
full story.

---

## Update 2026-10-12: third NaN root cause found (`inv_var_weight`), two new
## environment features (`reward_object_id`, periodic control-landmark
## reshuffling), cluster QoS/job-packing workaround, and a new campaign whose
## benchmark shows two regressions vs. the previous campaign

### Third NaN-corruption incident — real root cause this time

Gradient clipping (`f125ac4` above) reduced but did not eliminate the
corruption. Root cause, found by bisecting a corrupted run down to the exact
operation: **`inv_var_weight()` in `whittington_2020_utils.py` divides by
`sigmas**2` with no floor at all** — if a sigma estimate ever landed at or
near exactly 0, this produced `inf`/`nan` in one step, independent of
gradient magnitude (so clipping couldn't catch it — the corruption happens
*before* backward, not in the optimiser step). Fixed with
`torch.clamp(sigmas**2, min=1e-6)`. Combined with the gradient-clip guard and
a resumable last-known-good checkpoint (saved alongside the periodic eval,
so a corrupted run can resume from the last clean point instead of
restarting from episode 0), this is now defense-in-depth across three
independent failure points: the value-ratio normalisation (`7f67178`), the
optimiser step (`f125ac4`), and the loss-weighting division (this fix).

### New environment feature: `reward_object_id` — a dedicated, never-moving object that marks the reward location

`DiscreteObjectEnvironment` now accepts `reward_object_id` (env_kwargs). If
set, `generate_objects()` reserves the state nearest `reward_location`
*exclusively* for this object id, before any landmark/control-landmark/decoy
sampling runs (excluded from both `candidate_states` and `decoy_states` so
nothing else can land there). `reshuffle_control_landmarks()` also excludes
this reserved state from its own candidate pool, so the reward marker never
gets displaced by a reshuffle. Verified in isolation: object always at the
reward state, survives repeated reshuffles, exact expected counts with zero
stray "other" states. Rationale: give the agent one object identity that
*always* co-occurs with reward, independent of which value/control landmark
happens to be nearby — a cleaner, more direct reward-location cue than
relying on proximity-biased landmark placement alone.

### New environment feature: periodic control-landmark reshuffling

`DiscreteObjectEnvironment.reshuffle_control_landmarks()` + a
`control_shuffle_interval` parameter on `tem_training_loop()` — every N
episodes, control-landmark identities are redrawn (uniform, excluding
reserved landmark/reward states) while real (value-carrying) landmarks stay
fixed. Rationale: without this, a network could in principle learn to
decode value from a *specific* control landmark's fixed position/identity
rather than failing to decode value at all — periodic reshuffling forces any
"control landmark → no value" null result to hold across multiple physical
identities, not just one.

**Interval choice, revisited:** first implementation used
`control_shuffle_interval=100`, matching `rotate_environments`' own
per-100-episode rotation cadence. Asked directly whether 100 was the right
choice for "creating a bias to value-carrying objects that predict reward
position" — assessment was that 100 is too close to the rotation timescale
itself (shuffling and rotating at the same cadence conflates the two
effects), and it was lowered to **25** for the retrain below. This has NOT
been validated to be better, only reasoned to be more decorrelated from
rotation — see the benchmark finding below, which suggests the change may
have had unintended side effects.

### Cluster workflow: `--runs-per-node` job packing, to work around a hard QoS cap

`sacctmgr show qos mss_bigbatch` confirmed `MaxJobsPerUser=6` — a hard
concurrency cap regardless of how jobs are packed, separate from the
partition's general availability. Added `run_training_job_group()` to
`submit_experiments.py` plus a `--runs-per-node` CLI arg: packs N
(seed, condition) runs into one Slurm job as concurrent `subprocess.Popen`
siblings on the same node (scaling `slurm_cpus_per_task`/`slurm_mem`
accordingly), so more than 6 runs can be in flight at once without exceeding
the per-user job count. Also added `env["PYTHONUNBUFFERED"] = "1"` to every
subprocess launch — see the buffering fix below for why this matters.

### GPU vs CPU node detection — no Slurm-side metadata exists on this cluster

`scontrol show node` returns `(null)` for Gres/AvailableFeatures/
ActiveFeatures even on confirmed-GPU nodes — there is no way to query which
node is a GPU node before landing on it, and direct SSH from the login node
into a compute node is blocked (password auth, no key). The only reliable
check is empirical, from inside a running job: grep `run.log` for "TEM using
device: cuda" vs "cpu". This was unreliable at first because
`simulation_manager.py` opened `run.log`/`error.log` with Python's default
full block-buffering (`sys.stdout = open(run_log_path, "w")`) — the device
line could sit unflushed for minutes even on a perfectly healthy job, which
nearly caused a good GPU job to be cancelled prematurely by an automated
retry loop that saw an empty log and assumed it had landed on CPU. Fixed by
opening both logs with `buffering=1` (`ea497b0`). **Known GPU nodes:**
mscluster63, 71, 72, 79, 80, 81, 84. **Known CPU-only nodes:** mscluster44,
48, 57, 65, 83 (also recorded in the cluster-access memory file). Also
confirmed: `submit_experiments.py` must pass `--gres ''` on this cluster —
the default `--gres gpu:1` fails with "Invalid generic resource (gres)
specification" since GRES tracking isn't configured here at all.

### Campaign: `ctrlshuffle25_rewardobj_50k` — 5 seeds × 2 conditions, 50k episodes

Launched with `control_shuffle_interval=25` (down from 100) and
`reward_object_id=44` (new), otherwise matching the previous
`ctrlshuffle_50k` campaign's config (jobs 66867-66871, seeds 42/7/1/2/3,
`how_to_run.md`/cluster conventions unchanged). Benchmarked with the same
two probes used for every previous campaign (`accuracy_probe.py`,
1000-episode frozen-weight; `value_decode_probe.py`, the `reward_prediction`
project's `has_value` classifier methodology, pooled across all 5 seeds).

**Finding 1 — the reward-modulated accuracy advantage disappeared:**

| | baseline x_gt (mean) | RM x_gt (mean) | RM − baseline |
|---|---|---|---|
| `ctrlshuffle_50k` (old, interval=100, no reward obj) | 83.5% | 85.4% | **+1.9pp**, consistent across seeds |
| `ctrlshuffle25_rewardobj_50k` (new, interval=25, reward_object_id=44) | 84.1% | 83.6% | **−0.5pp**, inconsistent per seed |

Per-seed RM−baseline in the new campaign: seed42 +2.0pp, seed7 +0.3pp,
seed1 **−4.3pp**, seed2 −0.1pp, seed3 −0.3pp. x_p stayed saturated at 100%
in every run, so this is purely a grid/place (x_g/x_gt) effect. The old
campaign's clean, consistent RM advantage is gone — it's now close to a
coin-flip per seed, with seed 1 notably worse under RM.

**Finding 2 — shape/value dissociation got noisier, not cleaner:**

`has_value` decodability (pooled, held-out test, group-leakage-safe split):
accuracy=0.847, balanced_accuracy=0.822, AUC=0.920, permutation p=0.0033 —
still highly significant overall. But `bl_real` (false-positive rate — the
rate at which baseline's real-landmark population, which never carries
value, is wrongly decoded as value-carrying) rose from **0.113** (old
campaign) to **0.394 pooled** (new campaign), with large per-seed spread:
seed42=0.000, seed7=0.538, seed1=0.744, seed2=0.259, seed3=0.286.

**Interpretation (not yet validated by ablation):** both regressions moved
together when interval and reward-object were changed simultaneously, which
is suggestive but not conclusive that they share a cause — more frequent
control-landmark reshuffling (or its interaction with the new reward object)
may be destabilizing something that mattered for both the RM accuracy edge
and the clean shape/value dissociation. Which of the two changes
(`control_shuffle_interval` 100→25, vs. adding `reward_object_id`) is
responsible has not been isolated — would need a controlled ablation (e.g.
interval=25 without the reward object, or interval=100 with it) to pin down
before drawing further conclusions from this campaign's results.

---

## Update 2026-10-14: LC-inspired prediction-error gating (TEM-LC, TEM-R-LC)
## — mechanism, a reward-visitation sanity check, and a campaign that
## cleanly beats both prior campaigns on both metrics

### Mechanism: gate the Hebbian write by one-step sensory prediction error

New, independent, switchable module (`use_lc_gating`) alongside the existing
reward/value mechanism (which this does **not** touch) — inspired by locus
coeruleus norepinephrine signalling: scale the Hebbian memory write rate by
how surprising the current step is, instead of writing at a uniform rate.

- `ε_t = cross_entropy(x_logits[0], labels)` — reuses `Model.generative()`'s
  already-computed `x_p_logits` (`whittington_2020_model.py`'s
  `generative()`, called *before* `hebbian()` in `iteration()`), so this is
  exactly the same quantity as the existing `L_x_p` loss term, computed
  fresh and **detached** (a control signal, not something trained via
  backprop) — zero new learned parameters, no extra forward pass.
- `g(ε_t) = sigmoid(lc_beta * (ε_t − ε₀))`, `ε₀` a per-batch-slot running
  EMA of `ε_t` (`lc_eps0_tau`), threaded through `Iteration`/`prev_iter`
  (not a plain `Model` attribute) so periodic eval/probe replays — which
  call `agent.tem()` directly against the same `Model` instance — get their
  own independent `eps0` chain instead of corrupting the live training one.
  Resets per batch slot via the *existing* `init_walks()` hook (the same
  one `rotate_environments` already uses to mark a fresh environment
  instance), so "recent ε_t" means recent within this environment, not
  blended across a rotation.
- `Model.hebbian()` gained an independent `lc_scale` kwarg (alongside the
  pre-existing, still-unused `td_scale` kwarg — different mechanism,
  untouched), multiplied into `eta` the same way: `M(t) = λM(t−1) +
  η(t)·g(ε_t)·(p−p̂)(p+p̂)ᵀ`. Baseline's existing `η(t)` annealing schedule is
  preserved unchanged. `use_lc_gating=False` is a no-op by construction —
  verified, not assumed (see below).
- Config: `use_lc_gating`/`lc_beta`/`lc_eps0_tau` default `False`/`5.0`/
  `0.01` in `parameters.py`; CLI `--lc-gating`/`--lc-beta`/`--lc-eps0-tau`
  in `submit_experiments.py`; env vars `TEM_LC_GATING`/`TEM_LC_BETA`/
  `TEM_LC_EPS0_TAU`. Condition folders get a `_lc` suffix
  (`baseline_lc`/`reward_modulated_lc`), additive — existing `baseline`/
  `reward_modulated` names are unaffected when the flag is off.
- Per-step `ε_t`/gate logging: `lc_gating_log.npz` written at every existing
  eval checkpoint (`_tem_eval.py`'s `_flush_lc_log()`), reusing the same
  trailing-window slicing already used for place-cell rates.

### A real, pre-existing reproducibility gap found and fixed along the way

Verifying "gate off reproduces baseline exactly, same seed" surfaced that
this repo had **zero `torch.manual_seed` calls anywhere**, and — less
obviously — that `np.random.seed()` wasn't enough either:
`Model.init_trainable()` initialises `g_init`/`logsig_g_init` (trainable
grid-cell priors) via `scipy.stats.truncnorm.rvs()`, which draws from
**numpy's** global RNG, at agent-construction time — i.e. *before*
`tem_training_loop`'s own `random.seed`/`np.random.seed` calls ever run
(those only cover object-layout generation and the action sequence, both
inside `env.reset()`, which happens after construction). Fixed by seeding
all three RNG streams together at the earliest point in
`whittington_2020_run.py`, before agent/env construction. This is a general
fix, not LC-specific — it makes every future run's weight initialisation
actually reproducible under the same `TEM_SEED`, which was silently false
before.

### Verification (`verify_lc_gate_off.py`, `inspect_lc_gate_tuning.py`)

Both run the real training pipeline (via `submit_experiments.run_training_job`
as a subprocess, not a reimplementation), forced onto CPU to isolate the
check from CUDA's own non-determinism (a separate, pre-existing property of
the whole stack, not fixed here):
- Gate off, same seed, two separate runs → every tensor in
  `agent.tem.state_dict()` is `torch.equal` — **PASS**.
- Gate on, same seed → produces different weights (not silently inert) —
  **PASS**.
- 300-episode gate-only tuning run with the defaults (`lc_beta=5.0`,
  `lc_eps0_tau=0.01`): gate starts near-uninformative (episode 1, single-
  sample window, mean=0.5), and by episode 300 shows real position
  selectivity — mean gate 0.195, 70% of positions suppressed (<0.1,
  familiar/predictable), 14% strongly gated (>0.9, still surprising). Not
  saturated, not flat — defaults used as-is for the full campaign, no
  retuning needed.

### Does the agent's random walk actually reach the reward state often enough to matter?

Raised directly: the walk policy (`action_policy()`) is **uniform random
over 4 actions**, with zero reward-seeking bias — value/reward never
influences which direction the agent moves (this TEM implementation learns
value passively via TD on top of undirected exploration; it does not do
goal-directed action selection). So how often the agent's single `reward
state (nearest `reward_location`, which is also the dedicated
`reward_object_id`'s state) gets visited is purely a function of walk
length and arena coverage, not of what's been learned — worth checking
directly rather than assuming.

New diagnostic (`examples/agent_examples/check_reward_visits.py`): builds
env/agent from a saved `params.dict` but loads no trained weights and never
calls `agent.update()` (skips the TEM forward pass entirely, since the
random policy doesn't depend on it) — just runs `agent.batch_act()`/
`env.step()` in a loop, tallying per-env-slot visits to the reward state.
Results, 16 env slots, mixed arena sizes 64-121 states:

| scale | total steps/env | visits/env (min / mean / max) | zero-visit slots |
|---|---|---|---|
| probe (1000 episodes) | 20,000 | 126 / 269.8 / 387 | 0/16 |
| training (50,000 episodes) | 1,000,000 | 8,219 / 12,768.8 / 17,811 | 0/16 |

~1.3% of all steps land exactly on the reward state at both scales (higher
than the naive 1/n_states uniform-coverage baseline — plausible, since
`reward_location` sits closer to one corner than arena center, and bounded
random walks dwell disproportionately near boundaries). **Conclusion:
reward-state coverage is not a concern at either probe or training scale**
— the walk reaches it thousands of times over a full run.

### Campaign: `ctrlshuffle25_rewardobj_lc_50k` — same config as `ctrlshuffle25_rewardobj_50k`, LC gating added, 5 seeds

Deliberately did **not** resubmit `baseline`/`reward_modulated` (already
fully trained for all 5 seeds under the prior campaign) — only the 2 new
conditions (`baseline_lc`, `reward_modulated_lc`, 10 jobs) were submitted,
reusing the existing checkpoints for the other half of the 2×2 comparison.
Jobs 67402-67406, identical config otherwise (`control_shuffle_interval=25`,
`reward_object_id=44`, `decoy_object_id=30`, size-cycle `[8,9,10,11]`, 50k
episodes). All 10 runs finished clean (0 NaN/Inf params in every checkpoint).
One job (seed 1) landed on a CPU node and took ~29h instead of the ~17h the
4 GPU-node seeds took - left to finish naturally since by the time this was
noticed it was already past the halfway point (continuing was faster than
restarting fresh on a GPU node).

**Accuracy (x_gt, mean of last 500 episodes, all 5 seeds):**

| | baseline | reward_modulated | RM − baseline |
|---|---|---|---|
| `ctrlshuffle25_rewardobj_50k` (no LC) | 0.841 | 0.836 | **−0.5pp, inconsistent** (seed 1: −4.3pp) |
| `ctrlshuffle25_rewardobj_lc_50k` (LC) | 0.819 | 0.843 | **+2.4pp, positive on all 5 seeds** (42:+4.9, 7:+0.3, 1:+5.8, 2:+0.3, 3:+0.7) |

LC gating costs baseline ~2.2pp of raw accuracy (plausible: the gate
suppresses most Hebbian writes - only ~14-30% of positions get a strong
write per the tuning run - so less total memory signal accumulates when
there's no value mechanism giving that suppression anything useful to key
off). But `reward_modulated_lc`'s accuracy (84.3%) is essentially unchanged
or slightly *better* than non-LC `reward_modulated` (83.6%), and critically
the RM-vs-baseline gap becomes clean and consistent for the first time in
this line of campaigns - seed 1, the single worst outlier under non-LC
(RM **−4.3pp** worse than baseline there), becomes the single *best* seed
under LC (RM **+5.8pp** better).

**Value decodability (`has_value` classifier, pooled, held-out test) - LC beats both prior campaigns on every metric:**

| | `ctrlshuffle_50k` (oldest) | `ctrlshuffle25_rewardobj_50k` (no LC) | `ctrlshuffle25_rewardobj_lc_50k` (LC) |
|---|---|---|---|
| accuracy | 0.920 | 0.847 | **0.939** |
| balanced accuracy | 0.874 | 0.822 | **0.889** |
| AUC | 0.965 | 0.920 | **0.978** |
| `bl_real` pooled (false-positive rate, want ~0) | 0.113 | 0.394 | **0.044** |
| `bl_real` per-seed range | - | 0.000-0.744 | **0.000-0.103** |

Every per-seed `bl_real` rate is now low and consistent (seeds 42/7/1/2/3:
0.103/0.077/0.000/0.000/0.057) - none of the wild seed-to-seed spread the
non-LC campaign had.

### What "value decodability" is actually showing - clarifying a subtlety

The classifier distinguishes **`rm_real`** (the value-eligible landmarks in
the `reward_modulated_lc` run - `is_landmark==1` AND `value` explicitly
assigned, i.e. `V[landmark]` was injected into the compressed sensory code
`x_c` via the *original*, LC-untouched TEM-R mechanism) from three negative
groups: `rm_control` (same run, same kind of never-duplicated landmark
object, but one that never got a value assigned - a same-run, same-
condition contrast), and `bl_real`/`bl_control` (the `baseline_lc` run,
where `use_reward=False` means **no object of any kind ever gets a value
injected** - LC gating only changes the Hebbian learning-rate schedule, it
has no mechanism to write value into anything).

This means `bl_real`'s objects are, structurally, identical to `bl_control`'s
- both are just spatially-fixed, never-duplicated landmark-type objects in
a run where nothing carries value. The low `bl_real` rate (0.044 pooled) is
not a weak or surprising result - it is the **correct** answer, and it's the
most important number in the table: it confirms the classifier isn't just
keying off "this is a never-duplicated, reward-proximity-biased landmark"
shape/placement cue (which `bl_real` objects also have, despite carrying no
value) - if it were, `bl_real` would score as high as `rm_real` and the
whole measurement would be confounded. The fact that `rm_control` (same
run as `rm_real`, same object type, zero value) also scores ~0% reinforces
this from the other direction.

So: the agent did **not** spontaneously learn to treat objects as valuable
without ever being given a value signal - `reward_modulated_lc`'s `rm_real`
objects were explicitly given one (same mechanism as plain TEM-R, completely
untouched by this feature). What LC gating changed is how *cleanly and
consistently* that already-injected signal shows up in place-cell activity
across seeds, not whether a signal exists in the first place. The honest
reading is: gating the Hebbian write by prediction error appears to make the
memory system a more faithful, less noisy encoder of whatever value signal
it's actually given - not a mechanism that infers value where none was
---

## Update 2026-10-18: the no-shuffle campaign completed, then a long,
## mostly-humbling methodology investigation into what "value decodability"
## actually proves - several confounds found, one found to invalidate an
## earlier "decisive" result, and a narrower but more defensible finding
## survives

### No-shuffle campaign (`ctrlshuffle25_rewardobj_noshuffle_50k`) completed in full

All 5 seeds x both conditions now exist (the `reward_modulated` half was
trained after the original ablation only covered `baseline`) - all 10
checkpoints NaN/Inf-clean. This is the first **apples-to-apples** test of
the no-shuffle ablation, since the earlier landmark-vs-control proxy test
never had a real value-carrying condition to compare against.

**Accuracy (x_gt):** baseline mean 0.832, `reward_modulated` mean 0.820 -
driven almost entirely by a severe seed-1 outlier (**-14.3pp**, far worse
than any outlier seen in the shuffle-enabled campaigns); the other 4 seeds
average **+2.0pp**, the normal direction.

**Value-decode (`has_value` classifier):** accuracy=0.836, balanced
accuracy=0.876, AUC=0.960 - still highly significant, but the group
breakdown is qualitatively different from every shuffle-enabled campaign:
`rm_control`=0.168 and `bl_control`=0.194 are **not near-zero**, for the
first time ever (they were always ~0.000 in both the no-LC and LC
campaigns). Disabling reshuffling doesn't clean up `bl_real`'s false-positive
problem - it makes the classifier noisier across *all three* negative-class
groups, and critically `bl_real` (0.244) and `bl_control` (0.194) are now
close to each other instead of `bl_control` sitting at a clean 0% floor -
once reshuffling is off, the classifier can no longer reliably tell
real-landmark-type from control-landmark-type among baseline objects at
all. This argues *against* "permanence causes the original confound" -
removing reshuffling made things noisier, not cleaner, suggesting
reshuffling was doing something generically useful for representational
stability, not creating a confound that removing it would fix.

### A new tool: object-id-tagged frozen-weight probing

`tem_probe_eval_allenvs_objectid.py` (new script) - same frozen-weight,
all-environments probe as the existing `tem_probe_eval_allenvs.py`, but
also keeps the raw object id per state (0-9 = value landmark, 10-19 =
control landmark, by this project's fixed convention) instead of
collapsing straight to `is_landmark`/`is_control_landmark` flags. Needed
for every id-level analysis below. **Requires `agent_hyper` to be synced
locally alongside `agent`/`params.dict`/`whittington_2020_model.py`** -
this file was only synced for the no-shuffle campaign originally;
forgetting it for the no-LC campaign caused all 5 seeds to fail instantly
with `FileNotFoundError` before any real work happened (cheap mistake,
easy to miss since `accuracy_probe.py`/`value_decode_probe.py` don't need
this file at all - only the standalone all-envs probes do).

**Operational lesson learned the hard way: this GPU cannot run 2
concurrent `accuracy_probe.py`/probe-script processes.** Running 2 at once
dropped GPU compute utilization to 31% while VRAM sat at 96% (11.8/12.2
GiB) - a memory-pressure stall, not normal 50/50 time-sharing. Killing one
process immediately raised the survivor's utilization to 51% and dropped
VRAM to 58%. **All probe/accuracy-script runs in this pipeline must now be
one process at a time**, confirmed repeatedly after this finding.

### Finding 1 (solid): raw object identity alone is PERFECTLY decodable - any arbitrary partition

`check_fake_identity_value.py` - fabricated a completely arbitrary split
of the 20 landmark+control ids (first 5 + last 5 = "fake-valuable", middle
10 = "not", deliberately straddling the real landmark/control boundary so
it can't be explained by that category either) on the no-shuffle
baseline's (zero value anywhere) place-cell activity. Result: **accuracy =
balanced accuracy = AUC = 1.000**, zero errors on 256 held-out states.
This is unconfounded by any cross-network comparison (single network,
single dataset, arbitrary label) and stands as solid: place-cell activity
encodes raw per-object identity so completely that *any* id partition is
trivially separable. Every "value decodability" number measured anywhere
in this project is upper-bounded by this - high separability between
`rm_real` and anything else was never strong evidence of value-detection
on its own.

### Finding 2 (retracted/narrowed): cross-network comparisons are confounded by generic network divergence, not value

Two follow-up tests initially looked like clean positive evidence of a
real value signal, and were presented that way - both turned out to be
compromised by the same root issue, caught by a sharp external critique
(credited - this section exists because of it):

- `check_identity_residualized_value.py`: per-object-id mean-centering the
  `has_value` classifier's features before refitting. Balanced accuracy
  barely dropped (0.952 -> 0.929) and `bl_real`'s false-positive rate
  *fell* (0.143 -> 0.057) - looked like value surviving identity-removal.
- `check_matched_pair_value.py`: `reward_modulated - baseline` activity
  differences at matched (seed, env_id, state_id) triples (layouts are
  byte-identical across conditions for a given seed - verified directly,
  zero `is_landmark`/`is_control_landmark` mismatches). Scored 0.80-0.86
  balanced accuracy across all 3 campaigns telling real-landmark diffs
  from control-landmark diffs - presented as "decisive" at the time.

**Both are flawed for the same reason**: `baseline` and `reward_modulated`
(or the pooled-and-residualized equivalent) are two **separately trained
networks**. After 50k episodes of different gradient signals, their
weights diverge for reasons having nothing to do with value - and since
real landmarks are *always* ids 0-9 and controls *always* 10-19 (the exact
invariant Finding 1 already proved is trivially decodable), any
id-correlated divergence between two differently-trained networks can
masquerade as a value signal. Neither test controlled for this.

**Directly confirmed by `check_matched_pair_null.py`**: ran the identical
matched-pair test on a pairing where *neither side ever carries value* -
no-LC campaign's `baseline` vs LC campaign's `baseline_lc`, same seed
(layout match verified first, zero mismatches, all 5 seeds). Result:
**balanced accuracy 0.879, AUC 0.971 - higher than every one of the
"value" comparisons (0.807-0.862).** A zero-value-both-sides pairing
separates real from control landmarks at least as well as the pairing
that supposedly had value. This directly invalidates the matched-pair
test's "decisive" framing - its signal was never specific to value.

**Follow-up (`check_landmark_vs_control_condition_predict.py`)**: tried a
difference-in-differences design instead - predict *which condition*
(baseline or reward_modulated) produced a row, once using only landmark
rows, once using only control rows, same two networks both times, so any
value-specific effect should show landmarks scoring higher than controls.
Result across all 3 campaigns: **controls score slightly *higher* than
landmarks every time** (gaps of -0.006 to -0.025) - the opposite of the
predicted direction. No evidence of landmark-specific extra divergence
beyond generic network noise.

**Lesson for any future test in this line of work**: a test is only valid
evidence of a value-specific signal if it (a) never compares two
separately-trained networks against each other, and (b) never varies by
the landmark/control id-range alone (since that range is perfectly, and
separately, decodable per Finding 1). Any design that does either is
measuring an unknown mixture of value and these two confounds, not value
cleanly.

### Finding 3 (survives, but narrower than first claimed): relative learned value is decodable within a single network, controlling for identity

`check_within_network_value_rank.py` - designed to avoid both confounds
above: uses *only* `rm_real` rows from the reward_modulated network (no
cross-network comparison), *only* the landmark id-range (no id-range
contrast), labels each landmark as above/below its own environment's
median learned `V(id)` (relative rank, controls for absolute V-scale
differences across environments/seeds), held out by *entire environment*
(an environment's 10 landmarks never split across train/test). Result:
**balanced accuracy 0.806-0.850 across all 3 campaigns**, all highly
significant.

**Five further control checks** (`check_value_rank_controls.py`, run on
the no-shuffle campaign, seeds 7/2/3 - the only ones with object-id data
for both conditions at the time) narrowed this considerably:

| check | balanced accuracy | significant? |
|---|---|---|
| object_id alone -> label (no place cells) | 0.600 | no (p=0.12) |
| distance-to-reward alone -> label | **0.860** | **yes** |
| distance+id -> label | 0.880 | yes |
| baseline's own features -> RM's label | 0.580 | no (p=0.093) |
| RM features -> id-residualized label | 0.890 | yes |
| baseline features -> id-residualized label | 0.630 | yes (p=0.0066) |
| leave-one-seed-out (RM features, original label) | 0.783 (mean of 3 folds) | - |

Raw id alone does *not* explain the result (rules out a fixed
per-id placement-order artifact). But **plain distance-to-reward alone -
one number, no network - predicts the label better than the place-cell
classifier does.** That substantially narrows the claim: place cells
partially succeeding is not surprising since they are position encoders
by definition, and distance mechanically determines TD value. The
informative part is Check 3: **baseline's own place cells (same
architecture, same positions, zero value ever) cannot predict RM's label
above chance**, even though they have the same position-encoding
machinery raw distance has. If this were pure shared position-leakage,
baseline should succeed too; it doesn't. Check 4 (residualizing away each
id's fixed tendency) still works for RM (0.890) but baseline is not
perfectly at chance either (0.630, p=0.0066) - a real gap exists (0.890 vs
0.630) but not a fully clean RM-significant/baseline-chance dichotomy.
Check 5 confirms the effect generalizes to entirely unseen,
independently-trained seeds (0.783 mean), weaker than the within-seed
number but clearly real.

**The defensible claim, narrowed from the original**: not "place cells
represent value" - rather, **"training with value makes relative
distance-to-reward more linearly decodable from place-cell activity than
training without value does, and this survives controlling for each id's
fixed tendency and generalizes across independently-trained seeds."**
Smaller than first claimed, but it actually survives this scrutiny, unlike
the matched-pair result above.

**Not yet run**: the same 5-check battery on the no-LC and LC campaigns
(needs object-id probing for them first - `agent_hyper` now synced,
command ready, paused mid-run twice at the user's request, cheap to
resume: `tem_probe_eval_allenvs_objectid.py` once per seed, sequential).

### Reframing the thesis question itself

A late but important reframing: decoding injected value from `rm_real` is
close to circular, since value is written directly into the compressed
sensory code `x_c` by construction - of course it's "decodable," that's
not an interesting finding on its own. **The actually interesting
question is whether a *zero-value* run's permanently-fixed objects
(landmarks) get mistaken for valuable** - i.e. exactly what `bl_real`
already measures - and whether that holds up whether measured from the
live training snapshot or an independent frozen-weight probe, comparing
shuffling-enabled (no-LC) against no-shuffling as the controlled
comparison. For the no-shuffle campaign we already have both measurements
(training snapshot: `bl_real`=0.244/`bl_control`=0.194; standalone probe:
`bl_real`=0.143/`bl_control`=0.000 - noticeably cleaner, likely because
the standalone probe averages over a much longer, cleaner walk than the
live-training trailing window does). The equivalent standalone-probe
measurement for the no-LC (shuffling-enabled) campaign has not been run
yet (`check_training_vs_probe_blreal.py` is written and ready; the probe
run itself was paused mid-sequence at the user's request, to resume
later).

### New diagnostic scripts this update (all in `examples/agent_examples/`)

- `tem_probe_eval_allenvs_objectid.py` - frozen-weight all-envs probe
  keeping raw `object_id` per state (needs `agent_hyper` synced).
- `check_fake_identity_value.py` - arbitrary id-partition decodability
  (Finding 1).
- `check_identity_residualized_value.py` - per-id mean-centered
  `has_value` classifier (Finding 2, superseded/narrowed).
- `check_classifier_transfer.py` / `check_identity_only_transfer.py` -
  cross-campaign classifier transfer tests (earlier in this investigation,
  before the cross-network confound was identified - read with that
  caveat in mind).
- `check_matched_pair_value.py` / `check_matched_pair_null.py` - the
  matched-pair test and the null pairing that invalidated its "decisive"
  framing (Finding 2).
- `check_landmark_vs_control_condition_predict.py` - difference-in-
  differences landmark-vs-control condition-prediction test (Finding 2).
- `check_within_network_value_rank.py` - the within-network relative-value
  rank test (Finding 3).
- `check_value_rank_controls.py` - the five control checks on Finding 3,
  parameterized by `--probe-dir`/`--seeds`/`--label` to run on any
  campaign's object-id probe data.
- `check_training_vs_probe_blreal.py` - training-snapshot vs
  standalone-probe `bl_real`/`bl_control` comparison (written, not yet
  run on no-LC).
- `check_noshuffle_decodability.py` - the original landmark-vs-control
  proxy test, superseded by the full apples-to-apples no-shuffle campaign
  result above once `reward_modulated` finished training.
provided.
