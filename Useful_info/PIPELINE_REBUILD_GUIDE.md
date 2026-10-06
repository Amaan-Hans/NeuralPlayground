# TEM-R Pipeline — Full Rebuild Guide

Chronological account of everything done in this project, in enough detail to
rebuild the whole pipeline from scratch. Organized in the order things
actually happened. Cross-reference `HANDOFF.md` for a state snapshot; this
file is the "how did we get here" build log.

---

## Phase 0 — Cluster access

```
ssh ahanslod@146.141.21.100
```
Key-based auth, no password. Login node: `mscluster-login.ms.wits.ac.za`.
Home: `/home-mscluster/ahanslod`.

**Gotcha:** a "REMOTE HOST IDENTIFICATION HAS CHANGED" warning on a new
machine is not necessarily a MITM — verify the presented ED25519 fingerprint
against a machine you already trust before accepting
(`SHA256:77fWlX+LWqXh0tOMT3s4JbMy57ZW25T8tba9hhV27DU` was confirmed genuine
here). Fix a stale local entry with `ssh-keygen -R 146.141.21.100`.

---

## Phase 1 — Repo and environment setup

1. Cloned the fork/branch with the TEM-R (landmark + reward-modulated value
   learning) work already implemented:
   ```
   cd /home-mscluster/ahanslod/Masters
   git clone https://github.com/Amaan-Hans/NeuralPlayground
   cd NeuralPlayground
   git checkout reward-as-observation
   ```
   (This branch already had: landmark placement, TD(0) value table, the
   `x_c` trailing-dimension value injection, `_tem_eval.py`'s eval/plotting
   infra, `run_full_experiment.py` driver, and `Useful_info/how_to_run.md`
   documenting all of it — none of that was built in this session, it was
   pulled in from the fork.)

2. Conda env `tem_env` already existed on the cluster with torch 2.7.1,
   numpy, pandas, matplotlib, scikit-image, scikit-learn, scipy, opencv,
   gymnasium — but its editable NeuralPlayground install pointed at a
   different (older) checkout. Repointed it:
   ```
   source /home-mscluster/ahanslod/miniconda3/etc/profile.d/conda.sh
   conda activate tem_env
   cd /home-mscluster/ahanslod/Masters/NeuralPlayground
   pip install -e .
   ```

3. **Slurm conventions learned the hard way — follow these from the start:**
   - Use `bigbatch` partition (14 cpus/node, 3-day max, QoS `mss_bigbatch`
     caps 6 concurrent jobs/user, 12 nodes/user).
   - **Never run anything compute-heavy directly via plain `ssh ... command`
     on the login node** — it's shared and will silently stall (discovered
     when a script hung indefinitely in D-state). Always `sbatch` or `srun
     --partition=bigbatch`.
   - **`/tmp` is local to whichever compute node a job lands on, not
     NFS-shared** — files written there by an `srun`/`sbatch` job are
     invisible from the login node afterward. Always write outputs under the
     repo's NFS-shared path (e.g. `experiments/.../datasets/`).
   - Not every `bigbatch` node has a working GPU — some show
     `torch.cuda.is_available() == False` or throw a CUDA init warning. Code
     that loads saved checkpoints must handle this (see Phase 3's
     `_CPUUnpickler`).
   - **`submit_experiments.py`'s default `--gres gpu:1` fails** with
     `sbatch: error: Invalid generic resource (gres) specification` on this
     cluster — `sinfo -o '%N %G'` doesn't reliably surface the right GRES
     string either (GPU presence isn't cleanly visible that way here, even
     though GPUs are available and get used once a job lands on a node).
     Fix: pass `--gres ''` to disable the GRES request entirely and just
     submit to `bigbatch` plain — matches how every prior submission in this
     project's history has worked (see the CPU-fallback point directly
     above). Don't waste time hunting for "the correct" GRES string; there
     isn't one needed.
   - Access: `ssh -i ~/.ssh/id_ed25519_cluster ahanslod@146.141.21.100` — the
     default `~/.ssh/id_ed25519` key does **not** work here, the `-i` flag is
     required or every attempt fails with `Permission denied (publickey,...)`.

---

## Phase 2 — First training runs, then a design change (decoy tiling)

1. Ran the existing `run_full_experiment.py` (random-policy version, both
   conditions `baseline`/`reward_modulated`) for seeds 42 and 123 via
   `sbatch` on `bigbatch`:
   ```
   cd examples/agent_examples
   python run_full_experiment.py --seed 42     # ~1h45m, both conditions in parallel subprocesses
   python run_full_experiment.py --seed 123
   ```
   Output: `experiments/random/seed_{42,123}/{baseline,reward_modulated}/`.

2. **Design change**: patched `neuralplayground/arenas/discritized_objects.py`'s
   `generate_objects()` to add a `decoy_object_id` option — when set, every
   non-landmark state is tiled with ONE fixed repeated object id instead of a
   random one, so the *only* spatially-distinguishing sensory information in
   the whole environment is the 10 landmark objects. Wired
   `DECOY_OBJECT_ID = 20` into `whittington_2020_run.py`'s
   `discrete_env_params`. Verified with a standalone unit test before
   trusting it (10 landmark states get unique ids 0-9, 90 decoy states all
   get id 20).

3. Re-ran seeds 42/123 with the new decoy-tiled design (same command as
   above) — this became the baseline design for everything after.

**Note on reproducibility:** torch's weight initialization is **not seeded**
anywhere in `whittington_2020_run.py` — only `TRAJECTORY_SEED` is (via
`random.seed`/`np.random.seed`, applied to environment layout and action
selection). So re-running with the same seed reproduces the identical
environment/landmark layout and trajectory, but **not** identical trained
weights. If you want byte-identical reruns, add `torch.manual_seed(seed)`
before model construction — this was flagged but never actually added.

---

## Phase 3 — Frozen-weight probes, env-0 only

Built `tem_probe_eval_2k.py` (new file, in `examples/agent_examples/`):
loads the final trained weights, walks env 0 under a random policy for 2000
episodes (40,000 steps) with weights fully frozen (`agent.update()` never
called), computing place/grid rate maps averaged over the **entire** walk
(not just `run_eval()`'s trailing-500-step window), processed in
memory-safe chunks (10 episodes/chunk, recurrent state carried across chunks
via `prev_iter`) to avoid GPU OOM — the codebase's existing
`run_multienv_probe()` already used this pattern for a proven reason (a
naive full-window single forward pass OOMs).

**Key fix baked in from the start of this script:** a custom
`_CPUUnpickler`/`_load_state_dict` that intercepts `torch.storage._load_from_bytes`
during unpickling and forces `map_location="cpu"` — needed because saved
checkpoints contain CUDA tensors and some cluster nodes have no working GPU;
plain `pd.read_pickle`/`torch.load` both fail there without this.

Ran for both seeds/conditions → `experiments/random/seed_{42,123}/{baseline,reward_modulated}/plots/episode_probe_2000/`.

Built env-0-only CSV datasets from this (and separately from the existing
training-checkpoint data at episodes 2000 and 5000) with columns `seed,
condition, [episode,] state_id, x, y, is_landmark, is_reward, value,
p_<freq>_<idx>...` — two column-width variants: `full440` (all 440 place
cells) and `subset150` (first 30 cells × 5 frequency modules, matching what
the plots visually show). ~12 CSVs total downloaded.

---

## Phase 4 — All-environment probes

Extended to all 16 environments (previously only env 0 was ever evaluated —
`run_eval()` is hardcoded env-0-only, a limitation of the base codebase, not
a choice made here).

1. `tem_probe_eval_allenvs.py` (new file): reuses the codebase's own
   `run_multienv_probe()` (already handles all 16 envs' varying arena sizes —
   10×10, 8×8, 10×10, 12×12 repeating — memory-safe by construction) rather
   than reimplementing anything. Adds per-env layout/value/reward-state
   computation and CSV assembly on top. Same `_CPUUnpickler` fallback,
   extended to also cover `agent_hyper` and `params.dict` (not just the
   state_dict) after a job failed on a non-GPU node with a CUDA tensor
   buried inside `agent_hyper`.
   ```
   python tem_probe_eval_allenvs.py --run-dir ../../experiments/random/seed_42 \
       --seed-label 42 --episodes 2000 \
       --out ../../experiments/random/datasets/tem_dataset_seed42_allenvs.csv
   ```
   Output: `tem_dataset_seed{42,123}_allenvs.csv`, shape (3264, 449) each
   (2 conditions × 1632 total states across all 16 envs).

2. Wanted the same all-envs data **at each training checkpoint**
   (1000/2000/3000/4000/5000), not just a post-hoc frozen probe. This is
   **not retroactively extractable** — only the final trained weights were
   ever saved (no per-checkpoint weight snapshots), and envs 1-15 were never
   evaluated during the original training runs. Required retraining with a
   modified eval function:
   - `_tem_eval_allenvs.py` (new file): `run_eval_combined()` calls the
     existing `run_eval()` unchanged (so every existing env-0 plot still
     gets produced), then additionally computes/saves an all-envs
     place-cell CSV at checkpoints 1000-5000, via
     `_compute_multienv_rates_chunked()`.
   - **First attempt used `compute_multienv_rates()` directly and
     CUDA-OOM'd** — that function does a single monolithic forward pass over
     the whole 500-step window × 16 envs at once, which blows GPU memory
     when both `baseline` and `reward_modulated` subprocesses hit their eval
     checkpoint simultaneously and share one GPU. Fixed by writing
     `_compute_multienv_rates_chunked()` (chunks of 50 steps, `prev_iter`
     carried across chunks) — **verified numerically identical (0.0 max
     abs diff) to the original unchunked function** via a live side-by-side
     test before trusting it for the real run.
   - `whittington_2020_run_checkpoints.py` (copy of `whittington_2020_run.py`,
     only the eval-function import changed) and
     `run_full_experiment_checkpoints.py` (copy of `run_full_experiment.py`,
     launches the `_checkpoints` script and writes to a **new**
     `experiments/random_checkpoints/` directory so the existing
     `experiments/random/` results are never overwritten).
   - Hit and fixed an NFS directory-creation race (two `sbatch` jobs both
     trying to `os.makedirs` a brand-new shared parent directory
     simultaneously from different compute nodes — resubmitting the failed
     one after the other succeeded fixed it trivially).
   - Successful run → `experiments/random_checkpoints/seed_{42,123}/`, then
     aggregated the 5×2=10 per-checkpoint CSVs per seed into
     `tem_dataset_seed{42,123}_allenvs_checkpoints.csv` (16320 rows, 450
     cols each).

---

## Phase 5 — Control landmarks (second design change)

Added 10 more never-duplicated, reward-biased-placed objects (ids 10-19),
placed with the *exact same* sampling logic as the 10 value landmarks
(ids 0-9) — but living **outside** the agent's own `n_landmarks` range, so
the existing `held_landmark`/TD gating (`obj_id < self.n_landmarks` inside
`Whittington2020.batch_act`) automatically never tracks them — **zero
agent-side code changes needed**, purely a consequence of that existing
gate. Purpose: give a downstream decoder more spatially-unique sensory
anchors without touching the value-learning signal at all.

1. `discritized_objects.py`: added `n_control_landmarks` (env kwarg,
   default 0). `generate_objects()` now draws `n_landmarks +
   n_control_landmarks` states from one combined weighted-without-replacement
   sample (mathematically identical value-landmark placement statistics to
   before when `n_control_landmarks == 0`, since drawing the first *k* of a
   longer without-replacement sequence from the same weighted pool is
   statistically identical to drawing *k* directly).
2. `whittington_2020_run_checkpoints.py`: added `N_CONTROL_LANDMARKS = 10`,
   wired into `discrete_env_params` only (**not** `agent_params` — that's
   what keeps the agent's own `n_landmarks` at 10). Bumped
   `DECOY_OBJECT_ID` from 20 to 30 to avoid colliding with the new id range.
3. `_tem_eval_allenvs.py`: added an `is_control_landmark` output column,
   distinct from `is_landmark` (which still means "value landmark" only).
4. **Verified with a live smoke test before the real retrain**: confirmed
   10 value states (ids 0-9) + 10 control states (ids 10-19) + 80 decoy
   states, zero overlap; confirmed `held_landmark` and `agent.td.V.shape`
   never see anything above id 9 after a real walk; confirmed the CSV
   correctly flags both landmark types with no overlap (160 rows each,
   10 states × 16 envs).
5. Retrained both seeds with checkpoints (same jobs/commands as Phase 4,
   same output structure) — this became the final "current" trained model
   set used for everything after.

---

## Phase 6 — Getting models and code onto a local machine

1. Downloaded model bundles (the only things needed to reload a trained
   agent: `agent`, `agent_hyper`, `params.dict`, `training_hist.dict`,
   `whittington_2020_model.py`) for both seeds × both conditions from the
   control-landmark run, to `Downloads\tem_trained_models\seed_{42,123}\{baseline,reward_modulated}\`.
2. `git clone -b reward-as-observation https://github.com/Amaan-Hans/NeuralPlayground`
   directly to a local machine (public repo, works without cluster
   involvement) → `Downloads\NeuralPlayground_reward-as-observation\`.
3. **Important**: that clone only has what's actually committed to git — the
   control-landmarks patch to `discritized_objects.py` and the new
   `_tem_eval_allenvs.py` / `whittington_2020_run_checkpoints.py` /
   `run_full_experiment_checkpoints.py` / `tem_probe_eval_2k.py` /
   `tem_probe_eval_allenvs.py` scripts were **never pushed to git** — they
   only ever existed as uncommitted files on the cluster's working copy.
   Had to manually copy all of these into the local clone afterward so it
   actually matches what trained the downloaded models.
4. Separately, a *different*, older local clone already existed at
   `TEM-local/TEM/NeuralPlayground/` — checked out from upstream
   `SainsburyWellcomeCentre/NeuralPlayground` on `main`, which predates all
   the TEM-R work entirely (no `use_reward`/`n_landmarks`/`td_alpha` support
   at all). Two files were copied into it early on as a one-off ask; it was
   never reconciled into a fully working local setup. **Don't confuse this
   with the correct clone in step 2** — this doc's Phase 6 clone is the one
   that actually matches the trained models.

---

## Phase 7 — Object-prediction probe (the "moving landmark" experiment)

The core question: does the model's own generative prediction of sensory
observations get disrupted when a value-carrying landmark is physically
relocated mid-walk, and does having value information (`reward_modulated`)
change how it recovers, compared to `baseline`?

**Mechanism used**: TEM's `Model.iteration()` computes THREE parallel
"generated observation" logits inside `generative()` — `x_p_logits`
(from *inferred*/posterior place code, already informed by the current
observation), `x_g_logits` (from *inferred* grid code, also posterior), and
`x_gt_logits` = **`x_logits[2]`** (from `g_gen`, the *predicted* grid state
computed purely from transitioning `g_prev` via the action taken — this one
never looks at the current step's actual observation at all). Only
`x_logits[2]` is a genuine "predict before seeing" signal; the other two are
reconstructions. Confirmed this by reading the actual `generative()`
source before building anything on top of it.

1. `tem_object_prediction_probe.py` (new file): loads the final trained
   weights, walks env 0 for 2000 episodes with frozen weights, and at
   episode 1500 swaps one non-reward value landmark's object row with a
   currently-decoy state's row (direct mutation of
   `env.environments[0].objects`) — the landmark's identity is unchanged,
   just its physical location; `held_landmark`/TD tracking (keyed by object
   id, not state) follows it automatically.
   - Tracks running `(hits, total)` counters per `(category, window)` —
     `category` ∈ {control, value_nonreward, value_reward, decoy},
     `window` ∈ {pre, post} — **not** storing per-step data (per explicit
     instruction to keep it batched/lightweight).
   - First real run: accidentally executed directly via `ssh` on the login
     node — stalled indefinitely in D-state. Killed it, re-ran via
     `srun --partition=bigbatch`. **Lesson reinforced again**: never run
     compute-heavy work outside `srun`/`sbatch` on `bigbatch`.
   - Ran for both seeds → `object_pred_seed{42,123}.csv` (aggregate
     summary only, at this point).

2. Extended the same script to **also** save the full raw per-step log
   (predicted vs actual object id at every one of the 40,000 steps × 2
   conditions), the env-0 object layout before/after the move, and the
   exact move details (which landmark, from/to which state and
   coordinates). Hit the `/tmp`-is-not-NFS-shared gotcha again during the
   smoke test (wrote to `/home-mscluster/ahanslod/smoketest/` instead) —
   reran on `bigbatch` for both seeds:
   ```
   python tem_object_prediction_probe.py --run-dir ../../experiments/random_checkpoints/seed_42 \
       --seed-label 42 --episodes 2000 --move-episode 1500 --chunk-episodes 10 \
       --out .../object_pred_seed42.csv \
       --out-raw .../object_pred_seed42_raw.csv \
       --out-layout .../object_pred_seed42_layout.csv \
       --out-moveinfo .../object_pred_seed42_moveinfo.csv
   ```
   Downloaded all 8 files (4 per seed) to `Downloads\`.

---

## Phase 8 — Local analysis and visualization (no cluster needed from here)

Everything past this point ran **locally** (local Python 3.12 + pandas
2.2.3 + matplotlib 3.10.1 were sufficient — no need to go back to the
cluster) directly on the downloaded raw CSVs.

1. **Confusion matrices**: row-normalized (`P(predicted | actual)`) heatmaps,
   one 2×2 grid (window × condition) per seed. Used the `dataviz` skill's
   validated palette (sequential blue for magnitude, categorical
   blue/orange for baseline/reward_modulated — the skill's slots 1 & 2,
   pre-validated for CVD-safety).

2. **"Encounters the object at its new location" analysis**: distribution
   of what's predicted when revisiting state 72 (the landmark's new home)
   post-move, plus a cumulative-accuracy-over-successive-visits plot
   (revealing the Hebbian memory — which updates online even with frozen
   network weights — gradually re-learning the association, visible as a
   rise-then-plateau curve).

3. **User caught a real methodological issue**: pre-move confusion toward
   the decoy object seemed unexplained ("nothing was moved yet"). Verified
   empirically (not just asserted) that this is a genuine cold-start effect:
   plotted rolling accuracy across the whole pre-move window and confirmed
   accuracy climbs from ~0% to a plateau over the first ~5,000-8,000 steps,
   because the Hebbian memory is built fresh online during the probe walk,
   not carried over from training — the aggregate pre-window number mixes
   this transient in with the settled period.

4. **"After landmark visit" analysis**: for every visit to the moved
   landmark (both windows), looked at prediction accuracy 1 and 2 steps
   later — a cleaner, more consistent baseline-vs-reward_modulated
   comparison (reward_modulated wins in 7/8 groups) than the state-72-specific
   version, since it pools every visit rather than one fixed location.

5. **Spatial move map**: the 10×10 grid colored by category
   (value/control/reward/decoy) with the landmark's old (circle) and new
   (square) positions connected by an arrow.

6. **10-before/10-after sample timelines**: dot-strip visuals, one dot per
   visit, green = correct / red = incorrect, direct-labeled with the
   predicted object id. Built for both the at-landmark prediction and the
   next-state prediction.
   - **Bug caught by the user**: the next-state version showed the same
     predicted id (`1`) marked both green and red at different points —
     correct behavior, confusing presentation. Unlike the at-landmark case
     (where "actual" is always `1` by construction), the next state's
     actual identity varies by visit (the agent walks to a different
     neighbor each time). Fixed by labeling both `P<predicted>` and
     `A<actual>` on incorrect dots, then applied the same labeling to the
     at-landmark plot for consistency.
   - Also hit two mechanical bugs while iterating: a Windows file-lock
     (regenerating a PNG that was open in the IDE — worked around with a
     `_v2` filename) and a `sharex` regression (dropped it in a rewrite,
     causing only the bottom subplot to get correct tick labels).

---

## Quick-reference: every file created (not committed to git upstream)

On the cluster, `examples/agent_examples/`:
- `tem_probe_eval_2k.py` — frozen env-0 probe, full-walk averaging
- `tem_probe_eval_allenvs.py` — frozen all-envs probe (place cells)
- `_tem_eval_allenvs.py` — training-time all-envs checkpoint eval extension
- `whittington_2020_run_checkpoints.py` — training script using the above
- `run_full_experiment_checkpoints.py` — driver for the above
- `tem_object_prediction_probe.py` — landmark-relocation object-prediction probe

On `neuralplayground/arenas/discritized_objects.py` (modifies existing file):
- `decoy_object_id` support (Phase 2)
- `n_control_landmarks` support (Phase 5)

Locally, generated during analysis (`Downloads\`):
- 12 env-0-only dataset CSVs (Phase 3)
- `tem_dataset_seed{42,123}_allenvs.csv` (Phase 4.1)
- `tem_dataset_seed{42,123}_allenvs_checkpoints.csv` (Phase 4.2)
- `object_pred_seed{42,123}*.csv` (summary/raw/layout/moveinfo, Phase 7)
- Various `.png` visualizations + their source `.csv`s (Phase 8)
- `tem_trained_models/` — the 4 model bundles (Phase 6)
- `NeuralPlayground_reward-as-observation/` — matching local code clone (Phase 6)
- `cluster_scripts/` — example `.sbatch` templates

---

## If rebuilding from zero, do it in this order

1. Clone `Amaan-Hans/NeuralPlayground` @ `reward-as-observation`, set up
   `tem_env`-equivalent conda environment, `pip install -e .`.
2. Apply the `decoy_object_id` patch to `discritized_objects.py` (Phase 2)
   — do this **before** the first real training run, since it changes the
   environment design.
3. Apply the `n_control_landmarks` patch on top (Phase 5) — same file,
   additive.
4. Write `_tem_eval_allenvs.py`, `whittington_2020_run_checkpoints.py`,
   `run_full_experiment_checkpoints.py` (Phase 4/5) — use the chunked
   `_compute_multienv_rates_chunked()` from the start, don't repeat the OOM
   mistake.
5. Train both seeds via `run_full_experiment_checkpoints.py --seed {42,123}`
   on `bigbatch` (`sbatch`, never bare `ssh`).
6. Write `tem_probe_eval_2k.py` / `tem_probe_eval_allenvs.py` if you want
   post-hoc frozen probes independent of training checkpoints (optional —
   the checkpoint data from step 5 may already be enough).
7. Write `tem_object_prediction_probe.py` (Phase 7) for the landmark-move
   experiment, remembering: `x_logits[2]` for genuine pre-observation
   prediction, robust `_CPUUnpickler` loading, NFS-shared output paths
   (never `/tmp`), running counters unless you specifically want the raw
   per-step log too.
8. Everything from Phase 8 onward is local post-processing — no cluster
   access needed once the raw CSVs are downloaded.
