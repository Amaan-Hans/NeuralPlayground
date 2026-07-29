This markdown details the proposal for my research and is a sort of guideline for the results we would try to achieve.

# Simulating Reward-Driven Behaviour in the Hippocampus
Extending the Tolman–Eichenbaum Machine (TEM) with reward mechanisms.
Amaan Hanslod (2541305), Wits CS & Applied Math. Supervisors: Dr Devon Jarvis, Dr Victoria Williams.

## Core Question
Can reward-modulated learning mechanisms reproduce reward-dependent hippocampal phenomena (place-field accumulation near goals, backward temporal shift of reward coding) within the TEM framework?

## Background (key facts only)
- **TEM** (Whittington et al. 2020): factorises experience into structural code `g` (MEC, attractor dynamics, path integration), sensory observation `x` (LEC), combined via fast Hebbian learning into conjunctive hippocampal code `p`. Reproduces place cells, grid cells, border cells, object-vector cells, generalization via role-filler binding (reuse structural templates across sensory contexts).
- **Successor Representation** (Stachenfeld et al. 2017): hippocampus encodes expected future state occupancy; eigenvectors resemble grid cells. Doesn't explain sensory integration.
- **Reward evidence gap in TEM**: place fields accumulate near reward (Hollup et al. 2001, annular water maze); some hippocampal cells encode position relative to reward, not absolute space (Gauthier & Tank 2018); reward-coding cells shift firing backward in time to earlier predictive states over learning (Yaghoubi et al. 2026, "backward temporal shift").
- **LC (locus coeruleus)** activity precedes/gates hippocampal place-cell reorganization during reward learning (Kaufman et al. 2020) — interpreted as prediction-error-gated plasticity.
- Two candidate mechanisms: (1) reward/value becomes part of sensory observation (BLA→LEC pathway motivates this, no claim BLA does TD learning); (2) neuromodulatory (LC) signal gates *when* plasticity happens based on sensory prediction error.

## Hypothesis
- TD-value-augmented observation (TEM-R) reproduces *some* reward effects (e.g., place-field accumulation) but is **insufficient alone** for the full backward temporal shift.
- LC-inspired prediction-error gating (TEM-LC) is **necessary** to reproduce the progressive backward shift of reward coding.

## Research Questions
- **RQ1**: Does value-based learning (TD value in observation) reproduce reward-dependent spatial reorganization (place-field accumulation)?
- **RQ2**: Can LC-inspired prediction-error gating reproduce the backward temporal shift of reward coding?
- **RQ3**: Does combining both produce the most biologically realistic result (interaction effect)?

## Model: 4 Conditions (ablation design), implemented in NeuralPlayground framework
| Condition | Value in obs? | LC gating? | Primary eval env |
|---|---|---|---|
| Baseline TEM | No | No | All (control) |
| TEM-R | Yes | No | Hollup |
| TEM-LC | No | Yes | Kaufman |
| TEM-R-LC | Yes | Yes | Yaghoubi |

### TEM-R (value-augmented observation)
- TD error: `δ_t = r_t + γV_{t+1} − V_t`
- Value update: `V_t ← V_t + α·δ_t`
- Augmented observation: `x*_t = [x_t, V_t]` — fed into TEM as normal sensory input.
- Rationale: TD backpropagates value to reward-predicting states even before reward occurs, so augmented obs encodes expected future significance directly in state representation.

### TEM-LC (prediction-error gated Hebbian plasticity)
- Sensory prediction error (cross-entropy): `ε_t = − Σ x_obs · log(x̂ᵖ_t)`, where `x̂ᵖ_t = f_x(W_tile^T · p_t)` is softmax prediction from conjunctive hippocampal state `p_t` (`W_tile` fixed dimension-matching matrix per original TEM; `f_x` contains learned scalar weight + MLP).
- Gate: `g(ε_t) = σ(β(ε_t − ε_0))`, `ε_0` = running mean/EMA of recent prediction errors (adaptive, not fixed threshold); `β` tuned against Kaufman benchmark.
- Modified Hebbian update: `M(t) = λ·M(t−1) + η(t)·g(ε_t)·(p − pˣ)(p + pˣ)ᵀ`
- `η(t)`: same annealing schedule as baseline TEM (0 → 0.5 over training) — keeps gate suppressed early when predictions are uniformly poor everywhere.

### TEM-R-LC (combined)
- Uses `x*_t = [x_t, V_t]` as observation AND gates Hebbian update by `g(ε_t)` as above. Value = *what* is represented; LC gate = *when* it's updated.

### Baseline TEM
- Unmodified TEM, no reward signal, unmodulated Hebbian update. Must first reproduce: positive gridness score (structural code), place-cell-like spatial info in `p` (vs shuffle), generalization of structural templates across novel sensory contexts.

## Evaluation Environments (NeuralPlayground, discrete graphs)
1. **Hollup env** (annular ring, fixed reward) — targets place-field accumulation near reward. Metrics: place-field density near reward vs uniform baseline; reward zone enrichment ratio (>1.0 = success); goal-referenced coding (tuning shifts when reward moves).
2. **Kaufman env** (linear track, reward moved mid-training: Reward A → Reward B) — targets LC gating / anticipatory plasticity. Metrics: gating signal peaks near *new* reward-approach zone, not old; decorrelation of gating signal from raw traversal frequency over training; place-field enrichment at new reward location.
3. **Yaghoubi env** (2D grid, free exploration then fixed cue→reward trajectory, simplified linear track version of delayed non-matching-to-location task) — targets backward temporal shift. Metrics: backward shift of peak unit activity across sessions (correlation < 5th percentile of shuffle = "backward-shifting unit"); reward-encoding mutual information decline over sessions; pre-reward-state encoding MI increase; asymmetry check (backward shift above chance, forward shift at chance).

All 4 conditions × all 3 environments × 10 random seeds, same training protocol.

### Cross-condition comparisons (directly address RQ1–3)
- Hollup enrichment ratio: Baseline vs TEM-R → isolates value-augmentation effect.
- Yaghoubi backward-shift metric: TEM-LC vs TEM-R-LC → tests whether value augmentation enhances LC-driven reorganization.

## Timeline (Jul 2026 – Feb 2027)
1. Setup NeuralPlayground + verify baseline TEM — 1 wk (14 Jul)
2. Implement TEM-R/LC/R-LC — 2 wk (21 Jul)
3. Run Baseline + TEM-R on Hollup — 2 wk (4 Aug)
4. Run TEM-LC on Kaufman — 2 wk (18 Aug)
5. Run TEM-R-LC on Yaghoubi — 2 wk (1 Sep)
6. Aggregate metrics, cross-condition stats — 2 wk (15 Sep)
7. Follow-up experiments/tuning — 2 wk (29 Sep)
8. Figures + biological comparison — 2 wk (13 Oct)
9. Draft thesis — 5 wk (27 Oct)
10. Revise per supervisor feedback — 3 wk (1 Dec)
11. Polish/format — 2 wk (22 Dec)
12. Reproducibility docs — 2 wk (5 Jan)
13. Buffer — 3 wk (19 Jan)
14. **Final submission — 11 Feb 2027**

## Deliverables
Modular reproducible codebase (4 conditions, TD value head as ablatable module) in NeuralPlayground; quantitative metrics + visualizations; cross-condition figures addressing RQ1–3; full dissertation.

## Key Risks & Mitigations
- **Training instability** from architectural mods → validate incrementally (TEM-R alone → TEM-LC alone → combined); prelim runs stable so far.
- **Compute cost** (4 conditions × 3 envs × 10 seeds) → priority order if constrained: TEM-R-LC/Yaghoubi > TEM-R/Hollup > TEM-LC/Kaufman.
- **TD value head miscalibration** (γ, learning rate) → tune against Hollup benchmark first; verify smooth value gradient (not all-or-nothing).
- **LC gate poorly calibrated early** (uniformly high ε everywhere → degenerates to unmodulated Hebbian rule) → adaptive sigmoid gate + annealed η(t) mitigates.
- **Simplified Yaghoubi env** may drop critical features (e.g. working-memory delay) needed for backward shift to emerge.
- **Unit-tracking across sessions** required for backward-shift metric; discontinuous reorganization could break this.
- Report mean ± SD across 10 seeds; only count phenomenon as reproduced if consistent across majority of seeds.

## Limitations (acknowledged)
- All 3 environments are simplified vs original paradigms (proof-of-concept, not faithful replication).
- LC mechanism is a scalar computational abstraction, not biologically detailed (ignores NA/DA co-release, D1/D5 receptor specifics, inhibitory circuits).
- No behavior-driven trajectory changes (agent path is fixed, not reward-guided) — model isolates architectural effects from behavioral-sampling effects.
- Only 3 target phenomena evaluated; other reward-related hippocampal effects (reverse replay, goal-referenced remapping, reward-history encoding) out of scope.

## Notable predicted generalization property
Because value is bound to object identity (not just spatial position), value should transfer when a reward-predicting object appears in a new structural context. Testable prediction: disrupting object identity while preserving spatial structure should impair value transfer; disrupting spatial structure while preserving object identity should not.

## Key References (for citation if needed)
- Whittington et al. 2020, *Cell* — TEM.
- Stachenfeld et al. 2017, *Nat Neurosci* — successor representation.
- Hollup et al. 2001, *J Neurosci* — annular maze place-field accumulation.
- Gauthier & Tank 2018, *Neuron* — reward-referenced coding.
- Yaghoubi et al. 2026, *Nature* — backward temporal shift of reward coding.
- Kaufman, Geiller, Losonczy 2020, *Neuron* — LC role in CA1 reorganization.
- Sara & Bouret 2012, *Neuron* — LC arousal/cognition.
- Domingué et al. 2024, bioRxiv — NeuralPlayground framework.
- Sargolini et al. 2006, *Science* — gridness score methodology.
