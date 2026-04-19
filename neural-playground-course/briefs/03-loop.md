# Module 3: The Heartbeat Loop

### Teaching Arc
- **Metaphor:** A heartbeat has four phases: squeeze, pause, refill, relax. The training loop has four phases too: observe, act, step, update. Miss one and the whole system stops. Once you internalize this four-beat rhythm, you can read any RL codebase in the world.
- **Opening hook:** "There is one piece of code that runs tens of thousands of times in every simulation. It is four lines long. Everything else in the framework exists to support those four lines."
- **Key insight:** obs → act → step → update is the universal rhythm of reinforcement learning. The brain model reads its position (obs), decides where to move (act), the arena executes that move (step), and the brain model learns from what happened (update).
- **"Why should I care?":** When you ask AI to add a feature to this simulation — logging, visualization, early stopping — you need to know where in this loop to put it. "Add it after agent.update()" is a precise instruction. "Add it somewhere" is not.

### Code Snippets (pre-extracted)

**The core training loop — backend/training_loops.py (lines 29-41):**
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

**The act method (base agent) — agents/agent_core.py (lines 77-101):**
```python
def act(self, obs):
    if len(obs) == 0:
        action = None
    else:
        action = np.random.normal(scale=self.agent_step_size, size=(2,))

    self.obs_history.append(obs)
    if len(self.obs_history) >= self.obs_hist_length:
        self.obs_history.pop(0)

    return action
```

**The update method (base agent) — agents/agent_core.py (lines 103-105):**
```python
def update(self):
    """Update model parameters."""
    return None
```

### Interactive Elements

- [x] **Code↔English translation** — use the training loop snippet. Line-by-line:
  - `obs, state = env.reset()` → "Set up the arena from scratch — randomize the agent's starting position and clear all history"
  - `obs = obs[:2]` → "Keep only the x, y coordinates (ignore extra data like head direction for now)"
  - `action = agent.act(obs)` → "Ask the brain model: given your current position, which direction do you want to move?"
  - `obs, state, reward = env.step(action)` → "Execute that movement in the arena — check for walls, update position, return the new position"
  - `update_output = agent.update()` → "Let the brain model learn from what just happened — adjust its internal connection weights"
  - `training_hist.append(update_output)` → "Log whatever the update returned (weight changes, loss values, etc.) for later analysis"
- [x] **Data flow animation** — 3 actors: Agent, Arena, Backend. Steps:
  1. Backend: "Start: Arena resets, returns initial position [0.0, 0.0]"
  2. Arena is highlighted: "Arena initializes: random start position, walls loaded"
  3. Agent is highlighted: "Agent receives observation: [0.12, -0.34]"
  4. Packet from Agent to Arena: "Agent sends action: move in direction [0.8, 0.2]"
  5. Arena is highlighted: "Arena validates move (no wall collision), updates position"
  6. Packet from Arena to Agent: "Arena returns: new obs [0.18, -0.26], reward 0"
  7. Agent is highlighted: "Agent runs update() — adjusts weights based on what it experienced"
  8. Backend: "One heartbeat complete. Repeat 9,999 more times."
- [x] **Quiz** — 3 scenario questions:
  1. "You want to log the agent's position every 100 steps. Where in the loop do you add the logging code?" (Answer: after env.step(action), before agent.update() — you want the new position but not to delay learning)
  2. "The simulation is slow. A colleague suggests caching the last 10 observations so update() can use them all at once. Which variable already does this?" (Answer: obs_history in the agent — it stores past observations exactly for this purpose)
  3. Debugging: "Your agent keeps walking through walls. Which function should you investigate first?" (Answer: env.step() — specifically its validate_action() method which handles wall collision detection)
- [x] **Glossary tooltips** — reinforcement learning, observation space, action space, reward signal, weight update, gradient, step size, training loop, episode, n_steps

### Reference Files to Read
- `references/interactive-elements.md` → "Code ↔ English Translation Blocks", "Message Flow / Data Flow Animation", "Multiple-Choice Quizzes", "Glossary Tooltips", "Numbered Step Cards"
- `references/content-philosophy.md` → always
- `references/gotchas.md` → always

### Connections
- **Previous module:** "Meet the Cast" — introduced the four actors and their roles
- **Next module:** "How Neurons Learn Space" — dives into what actually happens INSIDE agent.update() for the Weber2018 and Stachenfeld2018 models
- **Tone/style notes:** Accent color is teal (#2A7B9B). Module background: var(--color-bg). The heartbeat metaphor should feel satisfying — this is the "aha!" moment where the whole framework clicks. Emphasize that this same loop pattern appears in ALL reinforcement learning systems — knowing it is a superpower.
