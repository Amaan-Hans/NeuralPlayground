# Module 1: Your Brain Has a GPS

### Teaching Arc
- **Metaphor:** Your brain has a built-in GPS that fires like a grid of streetlights as you move through space. NeuralPlayground is the software simulator that lets scientists test theories about how that GPS is wired.
- **Opening hook:** "Somewhere inside your skull, specialized neurons fire in a perfect hexagonal grid as you move around a room — like invisible dots on a map. Scientists won a Nobel Prize for discovering this. NeuralPlayground is the code that lets researchers simulate and test theories about how these neurons work."
- **Key insight:** This is a research framework — it gives scientists a standardized way to run virtual experiments: place a simulated brain model in a virtual arena, let it explore, and compare its neural firing patterns to recordings from real rats.
- **"Why should I care?":** Understanding what this framework does lets you steer AI assistants when extending it — knowing the difference between an "arena" and an "experiment" class means you can describe modifications precisely rather than vaguely.

### Code Snippets (pre-extracted)

**The simplest usage pattern — from backend/training_loops.py (lines 29-41):**
```python
obs, state = env.reset()
training_hist = []
obs = obs[:2]
for j in range(round(n_steps)):
    # Observe to choose an action
    action = agent.act(obs)
    # Run environment for given action
    obs, state, reward = env.step(action)
    update_output = agent.update()
    training_hist.append(update_output)
    obs = obs[:2]
```

**Agent initialization — from agents/agent_core.py (lines 55-68):**
```python
def __init__(
    self,
    agent_name: str = "default_model",
    agent_step_size: float = 1.0,
    obs_hist_length: int = 1000,
    **mod_kwargs,
):
    self.agent_name = agent_name
    self.mod_kwargs = mod_kwargs
    self.obs_hist_length = obs_hist_length
    self.agent_step_size = agent_step_size
    self.metadata = {"mod_kwargs": mod_kwargs}
    self.obs_history = []
    self.global_steps = 0
```

### Interactive Elements

- [x] **Code↔English translation** — use the training loop snippet (lines 29-41). English: "Reset the arena. For each step: ask the brain model what to do, move the agent, let the brain learn, record what happened."
- [x] **Data flow animation** — 4 actors: Agent, Arena, Experiment, Backend. Steps: (1) Arena initializes a 2D space, (2) Agent is placed inside, (3) Agent observes its position, (4) Agent picks an action (move direction), (5) Arena executes the move, returns new position, (6) Agent updates its internal weights, (7) Backend logs the result
- [x] **Quiz** — 3 questions, architecture/scenario style:
  1. Scenario: "You want to test a new theory about how place cells form. Which step should you focus on?" (Answer: writing a new Agent subclass with a custom update() method)
  2. Scenario: "A colleague says the model produces beautiful grid patterns — how can you verify this is scientifically valid?" (Answer: compare against Experiment data using GridScorer, not just visual inspection)
  3. Architecture: "Why does the framework separate Agents from Arenas?" (Answer: so the same brain model can be tested in multiple environments without rewriting code)
- [x] **Glossary tooltips** — grid cells, place cells, hippocampus, entorhinal cortex, neural recordings, firing rate, rate map, Python package/library, framework

### Reference Files to Read
- `references/interactive-elements.md` → "Code ↔ English Translation Blocks", "Message Flow / Data Flow Animation", "Multiple-Choice Quizzes", "Glossary Tooltips", "Numbered Step Cards"
- `references/content-philosophy.md` → always
- `references/gotchas.md` → always

### Connections
- **Previous module:** None (this is first)
- **Next module:** "Meet the Cast" — introduces the three main actors (Agent, Arena, Experiment) in detail
- **Tone/style notes:** Accent color is teal (#2A7B9B). Course is for vibe coders with zero neuroscience background. Be enthusiastic and accessible — this research won a Nobel Prize, so open with that hook. Module background: var(--color-bg) = warm off-white.
