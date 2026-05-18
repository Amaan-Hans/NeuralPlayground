# Results Interpretation: LC-Inspired Reward-Modulated TEM

**Experiment date:** May 2026
**Training:** 10 000 episodes × 20 steps/episode = 200 000 steps per condition
**Conditions:** Baseline TEM (no reward) vs Reward-modulated TEM (TD-gated Hebbian)
**Environment:** 16 parallel DiscreteObjectEnvironments, env 0 = 10×10 grid, start [0,0], reward [3,3]
**Analysis based on:** `tem_probe_eval.py` — frozen-weight walk, last 500 steps used for rate maps

---

## What each plot shows and what it tells us

---

### `population_activity_baseline.png` / `population_activity_reward_modulated.png`

**What it is:** Mean firing across all 400 place cells per grid state, plotted as a heatmap on the 10×10 arena at each 1000-episode checkpoint. Cyan star = reward location (nearest grid state to [3.0, 3.0] = state 77 at [2.5, 2.5]).

**What we see:**

*Baseline:* Early episodes (1000–2000) show high uniform activity across the whole grid (bright yellow — all states active). Over training the overall scale drops and activity becomes patchier, with some black (unvisited/low-activity) states appearing. The map becomes increasingly dark and diffuse — the population average flattens as individual cells specialise and no longer all fire everywhere.

*Reward-modulated:* Early episodes look similar to baseline. A notable bright flash appears at episode 3000 — a concentrated hot spot in the left-centre of the grid, far from the reward. This is a transient reorganisation event. By mid-training (ep 4000 onwards) the pattern converges to the same dark/diffuse state as baseline. At the endpoint both conditions look nearly identical.

**Interpretation:** The population-level representation converges to a similar diffuse state in both conditions, suggesting the reward gating does not produce a sustained population-level concentration of activity near the reward. The ep 3000 transient in the reward condition may reflect the model adjusting to the TD gate after the pretrain period (50 episodes is short).

---

### `value_correlation.png`

**What it is:** Pearson r between mean place cell activity per state (averaged across all 400 cells) and V(s) for env 0, at each checkpoint. Reward condition only — baseline has no V table.

**What we see:** The correlation oscillates with no clean trend. It starts moderately positive (~0.30 at ep 1000), drops sharply to –0.5 at episode 4000, recovers to ~+0.22 at ep 6000, then decays to near zero by ep 10000. There is no monotonic rise that would indicate cells becoming progressively more predictive of future reward.

**Interpretation:** The place cell population is not consistently encoding value. The strong negative correlation at ep 4000 is notable — place cells are most active in the *lowest*-value states at that checkpoint. This coincides with the hot spot seen in the population activity map at ep 3000–4000, suggesting a transient reorganisation where the model over-represents a low-value region. The ep 9000 peak (+0.37) is the highest correlation observed but is followed by collapse at ep 10000, suggesting instability rather than convergence.

**Caveat:** This metric averages over all 400 cells. Individual cells could still be value-predictive while others cancel them out in the mean. A cell-by-cell analysis would be more sensitive.

---

### `peak_distance_from_reward.png`

**What it is:** For each of the 400 place cells, find the state with highest average activation (peak firing location), compute its Euclidean distance to the reward location. Plot mean and median of those 400 distances at each checkpoint, for both conditions.

**What we see:** Both conditions track each other closely throughout training, both oscillating between ~5 and ~7 grid units. There is no sustained divergence — the reward condition is not consistently lower (cells clustering near reward) or higher (backward shift). At episode 9000 both conditions dip simultaneously to ~4.3–4.5 units, which is the closest either gets to reward. By ep 10000 baseline rises to 6.1 while reward condition stays at ~5.7 — a small difference but not a clear trend.

**Interpretation:** Reward gating is not causing place cells to systematically reorganise their preferred firing locations relative to the reward. The two conditions are statistically indistinguishable from this metric across training.

---

### `peak_distance_hist_baseline.png` / `peak_distance_hist_reward_modulated.png`

**What it is:** Histogram of the 400 peak-firing distances at episode 1000 (left) vs episode 10000 (right) for each condition. Red dashed line = nearest grid state to reward (~0.71 units).

**What we see:**

*Baseline:* Episode 1000 has a relatively concentrated distribution peaking around distance 6. By episode 10000 the distribution spreads and flattens, though the peak remains around 7–8 units. Very few cells fire near the reward at either checkpoint.

*Reward-modulated:* Episode 1000 is broad with a large spike at the far right (~10 units — the corner of the grid diagonally opposite the reward). By episode 10000 a new cluster appears at distance ~1 (just past the reward dashed line), but the far-right spike at ~10 units persists and is still large. The distribution is bimodal at the endpoint.

**Interpretation:** The reward condition develops a subset of cells (~13 at ep 10000) with peaks near the reward — this is the expected signal — but simultaneously pushes a large number of cells (~15+) to the far corner. This bimodality suggests the reward gating is reorganising representations but not cleanly — it concentrates some cells near the reward while displacing others to the opposite corner rather than creating a smooth backward-propagating gradient.

**The near-reward cluster at ep 10000 in the reward condition is the strongest positive signal in these results.** It does not exist at ep 1000 and is not present in the baseline at either checkpoint, indicating it is reward-driven. However the far-corner displacement undermines the clean interpretation.

---

### `grid_scores.png`

**What it is:** Mean grid score (spatial autocorrelation hexagonal periodicity) across all grid cells for both conditions over training.

**What we see:** The plot is effectively empty — a single dot at zero for both conditions (only episode 10000 available from the probe; earlier checkpoints have no `g_rates.npy`).

**Interpretation:** Cannot assess grid cell development over training from this data. The single-point grid score of ~0 likely reflects the 10×10 grid being too coarse for the GridScorer autocorrelation algorithm to detect hexagonal structure — the scorer needs spatial resolution finer than the cell spacing to measure periodicity. Meaningful grid scores require either a finer-resolution environment or the full 16-environment analysis used in the original TEM paper.

---

## Key experimental caveats

### 1. Rate maps are from a frozen-weight probe, not live training
The ep 10000 rate maps were generated by loading saved weights and running a fresh 700-step frozen walk. They reflect what the trained model *would have* produced at the end of training, not what was captured during training. Earlier checkpoints (ep 1000–9000) were captured live during training with a 600-step eval window at each checkpoint.

### 2. `p_inf` is not a firing rate
Place cell activity is `p_inf` — the TEM model's inferred posterior mean of the place cell latent variable given current grid code and sensory input. It is high when the model is confident about its location. This is not directly analogous to a biological place cell's spike rate. In particular it can be high at states rarely visited if the model has a sharp representation there.

### 3. Single environment analysis
All plots use env 0 only (10×10 grid, limits [-5,5]). The agent trains across 16 environments simultaneously. The reward exists only in env 0 at [3,3]. The representations in envs 1–15 are not analysed here and may differ.

### 4. Reward location is not a grid state centre
The reward is specified at [3.0, 3.0] but the nearest grid state centre is at [2.5, 2.5] (state id 77, distance 0.71 units). `DIST_TO_REWARD` is computed to [3.0, 3.0] so the minimum possible distance in the histogram is ~0.71, not 0. The red dashed line marks this minimum.

### 5. Pretrain period may be too short
`n_pretrain_episodes = 50` gives the model only 50 episodes (1000 steps) of unmodulated exploration before reward gating activates. TEM likely needs more time to form stable structural (grid cell) representations before Hebbian gating is introduced. The ep 3000 transient in the reward condition may reflect instability from gating activating before the model has converged.

### 6. TD learning rate and discount factor
`td_alpha = 0.1`, `td_gamma = 0.9`. With sparse reward (only state 77 gives r=1) and gamma=0.9, value propagates backward only ~2–3 steps reliably within 10000 episodes. States far from the reward have near-zero V(s) throughout training, meaning the TD gate is only meaningfully active near the reward. This limits the backward shift to a small neighbourhood around [2.5, 2.5].

### 7. Peak firing metric is undirected
`np.argmax(p_rates, axis=0)` finds the state of highest average activation but discards direction. A cell with its peak in the corner opposite the reward has the same distance (~10 units) as a cell firing in a genuinely distant approach state. The metric cannot distinguish between cells that are reward-approach-path-predictive and cells that fire in irrelevant locations.

### 8. V table is rebuilt from scratch in the probe
The saved agent file does not store V(s). During the probe walk, V is re-initialised to zero and updated from the fresh 700-step walk. With only 700 steps, V(s) converges near the reward state but may not have propagated far. The `v_table.npy` for ep 10000 reflects this short re-learning, not the full 10000-episode V history.

---

## Summary verdict

The reward gating has a detectable effect — the reward condition develops a small cluster of cells with peaks near the reward by episode 10000, absent in baseline. However there is no clean backward shift, no monotonic value correlation, and the far-corner displacement suggests the gating is disrupting spatial coverage rather than refining it. The most likely causes are the short pretrain period and the limited TD propagation radius given the current hyperparameters. These should be tuned before drawing strong conclusions.
