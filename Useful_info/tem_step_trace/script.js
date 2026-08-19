/* TEM One-Step Trace — signal-rail step-through
 * Data below mirrors neuralplayground/agents/whittington_2020_extras/
 * whittington_2020_model.py::Model.iteration(), vanilla TEM (no value
 * mechanism / no v / no td_scale args). One "step" = one call to
 * iteration() = one environment step. A training rollout calls this
 * n_rollout (=20) times before a single gradient update.
 *
 * Shapes use symbolic dims from whittington_2020_parameters.py defaults:
 *   n_x=45, n_x_c=10, n_f=5
 *   n_g            = [30, 30, 24, 18, 18]
 *   n_g_subsampled = [10, 10, 8, 6, 6]
 *   n_p            = [100, 100, 80, 60, 60]   (Σ = 400)
 */

const STAGES = [
  {
    phase: "start",
    phaseLabel: "Inputs",
    node: "in",
    title: "What one step receives",
    fn: "iteration(x, locations, a_prev, M_prev, x_prev, g_prev)",
    inputs: [
      { name: "x", type: "float · (batch, 45)", desc: "One-hot sensory observation — which of the 45 objects the agent is touching right now." },
      { name: "locations", type: "list[dict] · batch", desc: "{id: state index, shiny: None} per env — ground-truth position bookkeeping, not fed into inference as a shortcut." },
      { name: "a_prev", type: "int index (or None) · batch", desc: "The action just taken, one of 5 discrete moves. None marks the very first step of a walk." },
      { name: "M_prev", type: "float · (batch, Σn_p, Σn_p) × 1–2", desc: "Hebbian associative memory carried over from the previous step — one matrix (generative), optionally a second (inference)." },
      { name: "x_prev", type: "list[5] float · (batch, 10)", desc: "Last step's temporally-filtered sensory code — carries the EMA state forward." },
      { name: "g_prev", type: "list[5] float · (batch, n_g[f])", desc: "Last step's inferred grid code." },
    ],
    outputs: [],
    text: "Everything downstream is computed from just these six things. Nothing else leaks in — no privileged access to the true position beyond what a normal agent would have.",
  },
  {
    phase: "transition",
    phaseLabel: "Transition",
    node: "gen_g",
    title: "Predict from movement alone",
    fn: "gen_g(a_prev, g_prev, locations) → gt_gen, gt_inf",
    inputs: [
      { name: "a_prev", type: "int index · batch" },
      { name: "g_prev", type: "list[5] float · (batch, n_g[f])" },
    ],
    outputs: [
      { name: "gt_gen", type: "list[5] float · (batch, n_g[f])", desc: "Path-integrated grid code, used later by the generative pathway." },
      { name: "gt_inf", type: "(mean, σ) lists", desc: "Same prediction, packaged as a prior — a starting guess the inference pathway will correct." },
    ],
    text: "Before sensing anything new, TEM predicts where it should be from the action alone — eyes closed, trusting proprioception. A small MLP (f_mu_g_path) maps the previous grid code plus the action into a step, added to g_prev. This is shared machinery: both the generative and inference pathways start from it.",
  },
  {
    phase: "inference",
    phaseLabel: "Inference · bottom-up",
    node: "f_c",
    title: "Compress the observation",
    fn: "f_c(x) → x_c",
    inputs: [{ name: "x", type: "float · (batch, 45)", desc: "One-hot" }],
    outputs: [{ name: "x_c", type: "float · (batch, 10)", desc: "Two-hot compressed code" }],
    text: "The 45-way one-hot identity is compressed to a fixed 10-dim two-hot code via argmax + a static lookup table — not learned, just a deterministic re-encoding that gives every object a shorter signature.",
  },
  {
    phase: "inference",
    phaseLabel: "Inference · bottom-up",
    node: "x_prev2x",
    title: "Filter through time",
    fn: "x_prev2x(x_prev, x_c) → x_f",
    inputs: [
      { name: "x_prev", type: "list[5] · (batch, 10)" },
      { name: "x_c", type: "float · (batch, 10)" },
    ],
    outputs: [{ name: "x_f", type: "list[5] · (batch, 10)", desc: "One filtered code per frequency module" }],
    text: "Blends the new compressed code with the previous one via an exponential moving average — a different, learned decay rate per frequency module. Fast modules track new input almost instantly; slow modules drift gradually. This is TEM's analogue of grid modules operating at different temporal scales.",
  },
  {
    phase: "inference",
    phaseLabel: "Inference · bottom-up",
    node: "x2x_",
    title: "Normalise and tile",
    fn: "x2x_(x_f) → x_",
    inputs: [{ name: "x_f", type: "list[5] · (batch, 10)" }],
    outputs: [{ name: "x_", type: "list[5] · (batch, n_g_subsampled[f]×10)", desc: "Tiled sensory code, ready to pair with every grid dimension" }],
    text: "Zero-means and ReLUs the filtered code, reweights it, then tiles it out. This tiling is the setup for the conjunction two steps from now — it repeats the sensory vector once per grid dimension so an ordinary element-wise multiply later produces every (grid, sensory) pairing.",
  },
  {
    phase: "inference",
    phaseLabel: "Inference · bottom-up",
    node: "attractor",
    title: "Recall a place from touch alone",
    fn: "attractor(x_, M_inf) → p_x",
    inputs: [
      { name: "x_", type: "list[5]" },
      { name: "M_inf", type: "float · (batch, Σn_p, Σn_p)", desc: "Inference memory (optional second matrix in M_prev)" },
    ],
    outputs: [{ name: "p_x", type: "list[5] float · (batch, n_p[f])", desc: "Memory-retrieved place code" }],
    text: "Pattern-completes a place code purely from the current sensory cue, iterating the Hebbian memory's attractor dynamics — \"given only what I'm touching, what does memory say my place code should be?\" No grid or movement information enters here at all.",
  },
  {
    phase: "inference",
    phaseLabel: "Inference · bottom-up",
    node: "inf_g",
    title: "Correct the prediction with memory",
    fn: "inf_g(p_x, gt_inf, x, locations) → g_inf",
    inputs: [
      { name: "p_x", type: "list[5]", desc: "From the attractor step" },
      { name: "gt_inf", type: "(mean, σ)", desc: "From the transition step" },
    ],
    outputs: [{ name: "g_inf", type: "list[5] float · (batch, n_g[f])", desc: "Final, corrected grid code" }],
    text: "Combines the movement-only prediction (gt_inf) with a grid estimate distilled from the memory-retrieved place code (p_x), weighted by each one's own uncertainty — precision-weighted, like a Kalman filter. Whichever source is more confident wins. This is the position estimate everything else now builds on.",
  },
  {
    phase: "inference",
    phaseLabel: "Inference · bottom-up",
    node: "g2g_",
    title: "Reshape for the conjunction",
    fn: "g2g_(g_inf) → g_",
    inputs: [{ name: "g_inf", type: "list[5]" }],
    outputs: [{ name: "g_", type: "list[5] · (batch, n_g_subsampled[f]×10)" }],
    text: "Downsamples and retiles the corrected grid code, mirroring what x2x_ did on the sensory side, so the two are shaped identically for what happens next.",
  },
  {
    phase: "inference",
    phaseLabel: "Inference · bottom-up",
    node: "inf_p",
    title: "Bind position to identity",
    fn: "inf_p(x_, g_) → p_inf",
    inputs: [
      { name: "x_", type: "list[5]", desc: "Tiled sensory code" },
      { name: "g_", type: "list[5]", desc: "Tiled grid code" },
    ],
    outputs: [{ name: "p_inf", type: "list[5] float · (batch, n_p[f])", desc: "n_p = [100,100,80,60,60] — the place-cell population" }],
    text: "The conjunction. Grid code and sensory code are multiplied element-wise on their pre-tiled forms, so every grid dimension pairs with every sensory dimension. A cell only fires when both its grid phase and its sensory identity are simultaneously active — that coincidence-detection is what makes this a place code rather than a position code or an identity code alone.",
    highlight: true,
  },
  {
    phase: "generative",
    phaseLabel: "Generative · top-down",
    node: "gen_x·p",
    title: "Decode from what was just inferred",
    fn: "gen_x(p_inf[0]) → x_p, x_p_logits",
    inputs: [{ name: "p_inf[0]", type: "float · (batch, 100)", desc: "Highest-frequency module only" }],
    outputs: [
      { name: "x_p", type: "float · (batch, 45)", desc: "Softmax probabilities" },
      { name: "x_p_logits", type: "float · (batch, 45)" },
    ],
    text: "\"If my just-inferred place code is right, what should I be sensing?\" — decodes a predicted observation straight from p_inf. This is the tightest of the three decoding checks the loss will use, since it's asking the least of memory.",
  },
  {
    phase: "generative",
    phaseLabel: "Generative · top-down",
    node: "gen_p·g_inf",
    title: "Recall from position alone (corrected)",
    fn: "gen_p(g_inf, M_gen) → p_g_inf  →  gen_x → x_g",
    inputs: [
      { name: "g_inf", type: "list[5]", desc: "The corrected grid code" },
      { name: "M_gen", type: "float · (batch, Σn_p, Σn_p)" },
    ],
    outputs: [
      { name: "p_g_inf", type: "list[5]", desc: "Retrieved place code" },
      { name: "x_g, x_g_logits", type: "float · (batch, 45) each" },
    ],
    text: "Retrieves a place code from memory using only the corrected grid location — no sensory input at all — then decodes an observation from it. \"Given only where I now think I am, what does memory expect me to see?\"",
  },
  {
    phase: "generative",
    phaseLabel: "Generative · top-down",
    node: "gen_p·gt_gen",
    title: "Recall from position alone (uncorrected)",
    fn: "gen_p(gt_gen, M_gen) → p_g_gen  →  gen_x → x_gt",
    inputs: [
      { name: "gt_gen", type: "list[5]", desc: "The raw path-integration prediction" },
      { name: "M_gen", type: "float · (batch, Σn_p, Σn_p)" },
    ],
    outputs: [
      { name: "p_g_gen", type: "list[5]" },
      { name: "x_gt, x_gt_logits", type: "float · (batch, 45) each" },
    ],
    text: "Same retrieval, but from the uncorrected movement-only prediction (gt_gen) instead of g_inf. \"If I only ever trusted my movement and never let memory correct me, what would I expect to see?\" The gap between this and x_g is a training signal in its own right.",
  },
  {
    phase: "memory",
    phaseLabel: "Memory update",
    node: "hebbian",
    title: "Write the place code into memory",
    fn: "hebbian(M_prev, p_inf, p_gen) → M",
    inputs: [
      { name: "p_inf", type: "list[5]", desc: "Bottom-up (this step)" },
      { name: "p_gen", type: "list[5]", desc: "= p_g_inf, top-down (this step)" },
    ],
    outputs: [{ name: "M_new", type: "float · (batch, Σn_p, Σn_p)", desc: "Clamped, decayed, and updated" }],
    text: "An outer product of (p_inf + p_gen) and (p_inf − p_gen) is folded into memory, scaled by how much the bottom-up and top-down estimates disagreed. This is the only learning that happens outside of gradient descent — a fast, one-shot associative write, every single step.",
    dual: true,
  },
  {
    phase: "loss",
    phaseLabel: "Loss",
    node: "loss",
    title: "Score this step, for later",
    fn: "loss(gt_gen, p_gen, x_logits, x, g_inf, p_inf, p_inf_x) → L",
    inputs: [
      { name: "all of the above", type: "—", desc: "gt_gen, p_gen, the three x_logits, the true x, g_inf, p_inf, p_inf_x" },
    ],
    outputs: [
      { name: "L", type: "list of scalar tensors · batch", desc: "L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_reg_g, L_reg_p" },
    ],
    text: "Nothing here touches the model directly — these terms are what gradient descent uses afterward. Squared error between inferred and generated place codes (did bottom-up and top-down agree?); squared error against the pure-sensory retrieval; cross-entropy between each of the three decoded observations and the true one; L2/L1 regularisation on grid and place activity.",
    dual: true,
  },
];

const PHASE_META = {
  start:      { label: "Start" },
  transition: { label: "Transition" },
  inference:  { label: "Inference" },
  generative: { label: "Generative" },
  memory:     { label: "Memory" },
  loss:       { label: "Loss" },
};

let current = 0;

function buildRail() {
  const rail = document.getElementById("rail");
  rail.innerHTML = "";
  const track = document.createElement("div");
  track.className = "track";
  let lastPhase = null;

  STAGES.forEach((s, i) => {
    if (s.phase !== lastPhase) {
      // Phase boundary: a divider stands in for the wire, so the label
      // always lines up exactly with the node sequence it introduces
      // (both live in the same flex flow — no separate grid row to keep
      // in sync).
      const divider = document.createElement("div");
      divider.className = `divider phase-${s.phase}`;
      divider.textContent = PHASE_META[s.phase].label;
      track.appendChild(divider);
      lastPhase = s.phase;
    } else {
      const wire = document.createElement("div");
      wire.className = `wire phase-${s.phase}`;
      wire.id = `wire-${i}`;
      track.appendChild(wire);
    }
    const node = document.createElement("button");
    node.className = `node phase-${s.phase}`;
    node.id = `node-${i}`;
    node.type = "button";
    node.setAttribute("aria-label", `Step ${i}: ${s.title}`);
    node.innerHTML = `<span class="node-index">${i}</span><span class="node-tag">${s.node}</span>`;
    node.addEventListener("click", () => goTo(i));
    track.appendChild(node);
  });
  rail.appendChild(track);
}

function renderDetail() {
  const s = STAGES[current];
  const meta = PHASE_META[s.phase];

  document.getElementById("step-counter").textContent = `Step ${current} / ${STAGES.length - 1}`;
  const badge = document.getElementById("phase-badge");
  badge.textContent = s.phaseLabel;
  badge.className = `badge phase-${s.phase}` + (s.dual ? " badge--dual" : "");

  document.getElementById("stage-title").textContent = s.title;
  document.getElementById("stage-fn").textContent = s.fn;

  const renderList = (el, items, emptyText) => {
    el.innerHTML = "";
    if (!items.length) {
      const li = document.createElement("li");
      li.className = "io-empty";
      li.textContent = emptyText;
      el.appendChild(li);
      return;
    }
    items.forEach((it) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="io-name">${it.name}</span><span class="io-type">${it.type}</span>${
        it.desc ? `<span class="io-desc">${it.desc}</span>` : ""
      }`;
      el.appendChild(li);
    });
  };

  renderList(document.getElementById("inputs-list"), s.inputs, "—");
  renderList(document.getElementById("outputs-list"), s.outputs, "Nothing returned yet — feeds the next stage");

  document.getElementById("stage-text").textContent = s.text;

  document.getElementById("prev-btn").disabled = current === 0;
  document.getElementById("next-btn").disabled = current === STAGES.length - 1;
}

function updateRailState() {
  STAGES.forEach((_, i) => {
    const node = document.getElementById(`node-${i}`);
    node.classList.toggle("is-current", i === current);
    node.classList.toggle("is-visited", i < current);
    node.classList.toggle("is-future", i > current);
    const wire = document.getElementById(`wire-${i}`);
    if (wire) wire.classList.toggle("is-drawn", i <= current);
  });
  const currentNode = document.getElementById(`node-${current}`);
  currentNode.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
}

function goTo(i) {
  current = Math.max(0, Math.min(STAGES.length - 1, i));
  renderDetail();
  updateRailState();
}

document.addEventListener("DOMContentLoaded", () => {
  buildRail();
  renderDetail();
  updateRailState();

  document.getElementById("prev-btn").addEventListener("click", () => goTo(current - 1));
  document.getElementById("next-btn").addEventListener("click", () => goTo(current + 1));

  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") goTo(current + 1);
    if (e.key === "ArrowLeft") goTo(current - 1);
  });
});
