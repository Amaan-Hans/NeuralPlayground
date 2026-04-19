# Module 4: How Neurons Learn Space

### Teaching Arc
- **Metaphor:** Hebbian plasticity (Weber2018) is like "neurons that fire together, wire together" — imagine two musicians: the more they play the same note at the same time, the stronger their musical bond becomes. The Successor Representation (Stachenfeld2018) is like a subway map — instead of just knowing where you ARE, the hippocampus builds a mental map of where you are LIKELY TO GO from each location.
- **Opening hook:** "When a real rat explores a room, specific neurons fire ONLY when it's in a specific location — like having a dedicated alarm for every spot in the room. Two competing theories explain how this emerges. Both are implemented in this codebase. Here is how they actually work in the code."
- **Key insight:** Weber2018 uses Hebbian learning (synaptic strength grows when pre- and post-synaptic neurons fire together). Stachenfeld2018 uses Temporal Difference learning to build a predictive map of expected future positions. Both produce grid-like or place-like firing patterns from nothing — they emerge from the algorithm.
- **"Why should I care?":** When you ask AI to add a third brain model to this framework, you need to understand the pattern these two follow: inherit from AgentCore, implement act() to return a direction, implement update() to do your learning algorithm. That is the contract.

### Code Snippets (pre-extracted)

**Weber2018 update method — agents/weber_2018.py (lines 375-419):**
```python
def update(self, exc_normalization: bool = True, pos: np.ndarray = None):
    if pos is None:
        pos = self.obs_history[-1]
    if len(pos) == 0:
        pass
    else:
        r_out = self.get_output_rates(pos)
        # Excitatory weights update (eq 2)
        delta_we = self.etaexc * self.get_rates(self.exc_cell_list, pos=pos) * r_out
        # Inhibitory weights update (eq 3)
        delta_wi = (
            self.etainh
            * self.get_rates(self.inh_cell_list, pos=pos)
            * (r_out - self.ro)
        )
        self.we = self.we + delta_we
        self.wi = self.wi + delta_wi
        if exc_normalization:
            self.we = (
                self.init_we_sum / np.sqrt(np.sum(self.we**2) + 1e-8) * self.we
            )
```

**Weber2018 output rate calculation — agents/weber_2018.py (lines 308-328):**
```python
def get_output_rates(self, pos: np.ndarray):
    exc_rates = self.get_rates(self.exc_cell_list, pos)
    inh_rates = self.get_rates(self.inh_cell_list, pos)

    r_out = self.we.T @ exc_rates - self.wi.T @ inh_rates
    r_out = np.clip(r_out, a_min=0, a_max=np.amax(r_out))
    return r_out
```

**Stachenfeld2018 obs_to_state — agents/stachenfeld_2018.py (lines 179-200):**
```python
def obs_to_state(self, pos: np.ndarray):
    diff = self.xy_combinations - pos[np.newaxis, ...]
    dist = np.sum(diff**2, axis=1)
    index = np.argmin(dist)
    curr_state = index
    return curr_state
```

### Interactive Elements

- [x] **Code↔English translation** — use Weber2018 update() snippet. Line-by-line:
  - `pos = self.obs_history[-1]` → "Get the agent current position from memory"
  - `r_out = self.get_output_rates(pos)` → "Calculate how strongly the output neuron fires at this position"
  - `delta_we = self.etaexc * self.get_rates(...) * r_out` → "Excitatory weight change = learning rate × input neuron activity × output neuron activity (neurons that fire together, wire together)"
  - `delta_wi = self.etainh * self.get_rates(...) * (r_out - self.ro)` → "Inhibitory weight change = learning rate × input × (output minus a target rate) — this prevents the network from going silent OR firing everywhere"
  - `self.we = self.we + delta_we` → "Apply the update — strengthen the connections"
  - `self.we = (self.init_we_sum / np.sqrt(...) * self.we)` → "Normalize so total weight strength stays constant — prevents runaway growth"
- [x] **Pattern cards** — 3 cards showing Weber2018 vs Stachenfeld2018 vs Whittington2020:
  - Weber2018: "Hebbian Plasticity — neurons that fire together wire together. Produces grid cells. ~4900 neurons."
  - Stachenfeld2018: "Successor Representation — builds a predictive map of expected future locations. Uses TD learning."
  - Whittington2020: "Tolman-Eichenbaum Machine — a graph neural network. Requires discrete environments. PyTorch-based."
- [x] **Quiz** — 3 questions (architecture + debugging):
  1. "After 10,000 training steps, Weber2018 produces a uniform grey firing map — no clear spatial structure. What is the most likely cause?" (Answer: the exc_normalization is preventing any weights from growing — or learning rate etaexc is too small)
  2. "You want to add a new brain model that uses reinforcement learning signals (reward) instead of just position. Which method signature MUST your class implement to be compatible with the framework?" (Answer: act(self, obs) and update(self) — these are the AgentCore contract)
  3. Tracing: "In Stachenfeld2018, when the agent moves to a new position, which function converts the raw (x,y) coordinate into an index the model can use?" (Answer: obs_to_state() — it finds the nearest discrete state from the continuous position)
- [x] **Glossary tooltips** — Hebbian learning, synapse, excitatory neuron, inhibitory neuron, weight, learning rate, temporal difference learning, successor representation, eigendecomposition, place cell, grid cell, neural network layer

### Reference Files to Read
- `references/interactive-elements.md` → "Code ↔ English Translation Blocks", "Pattern/Feature Cards", "Multiple-Choice Quizzes", "Glossary Tooltips", "Callout Boxes"
- `references/content-philosophy.md` → always
- `references/gotchas.md` → always

### Connections
- **Previous module:** "The Heartbeat Loop" — showed the 4-beat loop, introduced agent.update() as the learning step
- **Next module:** "Real Rat Data" — shows the Experiment classes that provide ground truth for comparing these models
- **Tone/style notes:** Accent color is teal (#2A7B9B). Module background: var(--color-bg-warm). Include one callout box: "Emergence is magic" — these grid patterns are NOT programmed in; they emerge spontaneously from the learning rules. This is genuinely exciting and worth highlighting. Use "neurons that fire together, wire together" as a callout.
