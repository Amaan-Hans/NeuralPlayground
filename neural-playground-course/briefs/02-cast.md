# Module 2: Meet the Cast

### Teaching Arc
- **Metaphor:** Think of a stage play. The Arena is the set — the physical space where everything happens. The Agent is the actor — it has a role (a brain model) and follows a script (algorithms). The Experiment is the script consultant — real recordings from actual rats that tell you whether the performance is accurate. The Backend is the director — it coordinates everyone and keeps the show running.
- **Opening hook:** "Every simulation has four players. Once you know their names and roles, you can tell an AI assistant exactly what to change, where to change it, and why — without getting lost in 5,000 lines of code."
- **Key insight:** These four roles are separated on purpose (separation of concerns). You can swap any actor for another without rewriting the rest. Weber2018 brain model in a circular arena? Easy. Weber2018 brain model in a rectangular arena? Just swap the arena. Same brain, different stage.
- **"Why should I care?":** Knowing the cast is the difference between "AI, add something to the code" vs "AI, add a new method to the Arena class that returns the distance from the agent to the nearest wall." The second instruction is ten times more likely to get you what you want.

### Code Snippets (pre-extracted)

**AgentCore class header — agents/agent_core.py (lines 21-53):**
```python
class AgentCore(object):
    """Abstract class for all EHC models.

    Attributes
    ----------
    agent_name : str
        Name of the specific instantiation of the agent class
    obs_history: list
        List of past observations while interacting with the environment
    global_steps: int
        Record of number of updates done on the weights

    Methods
    -------
    reset(self):
        Erase all memory from the model, initialize all relevant parameters
    act(self, obs):
        Given an observation, return an action following the policy
    update(self):
        Update model parameters, depends on the specific model
    save_agent(self, save_path: str, raw_object: bool = True):
        Save current state and information
    restore_agent(self, save_path: str):
        Restore saved agent
    """
```

**Simple2D arena class — arenas/simple2d.py (lines 14-28):**
```python
class Simple2D(Environment):
    """Methods
    ----------
    reset(self):
        Reset the environment variables
    step(self, action):
        Increment the global step count and move the agent
    plot_trajectory(self, history_data=None, ax=None):
        Plot the Trajectory of the agent in the environment
    validate_action(self, pre_state, action, new_state):
        Check if the new state is crossing any walls in the arena
    render(self, history_length=30):
        Render the environment live through iterations as in OpenAI gym
    """
```

### Interactive Elements

- [x] **Code↔English translation** — use AgentCore class header snippet. English: "This is the blueprint for all brain models. Every model must know how to: observe and store what it sees, choose an action, learn from experience, save/restore itself."
- [x] **Group chat animation** — actors: Agent (teal), Arena (amber), Experiment (plum), Backend (forest). Chat script:
  - Backend: "Time to run a simulation. Arena, set up the environment."
  - Arena: "Ready. I have a 1m × 1m square room with walls and a random starting position."
  - Backend: "Agent, you are placed in the arena. Here is your first observation: position [0.12, -0.34]"
  - Agent: "Got it. Based on my current weights, I will move in direction [0.8, 0.2]."
  - Arena: "Movement executed. New position: [0.18, -0.26]. No walls crossed. Reward: 0."
  - Agent: "I am learning from this step. Updating my weights now."
  - Backend: "Step 1 complete. Logging result. 9,999 steps remaining."
  - Experiment: "When you are done, compare your firing patterns against my rat recordings."
- [x] **Visual file tree** — show the neuralplayground/ directory with all four actor folders annotated
- [x] **Quiz** — 3 scenario questions:
  1. "A new paper describes a brain model that uses spiking neurons instead of rate coding. Where in NeuralPlayground would you add it?" (Answer: agents/ — create a new class inheriting from AgentCore)
  2. "You want to test an existing agent in a T-maze environment. Which file would you need to create?" (Answer: a new Arena subclass in arenas/)
  3. "After training, you want to check if the model produces realistic grid cells. Which component provides the ground truth to compare against?" (Answer: Experiment — it holds real rat neural recordings)
- [x] **Glossary tooltips** — abstract class, inheritance, subclass, method, attribute, observation, action space, reward, gymnasium, OpenAI Gym

### Reference Files to Read
- `references/interactive-elements.md` → "Group Chat Animation", "Code ↔ English Translation Blocks", "Visual File Tree", "Multiple-Choice Quizzes", "Glossary Tooltips", "Icon-Label Rows"
- `references/content-philosophy.md` → always
- `references/gotchas.md` → always

### Connections
- **Previous module:** "Your Brain Has a GPS" — established what NeuralPlayground does and showed the high-level training loop
- **Next module:** "The Heartbeat Loop" — zooms into exactly what happens at each step of the loop
- **Tone/style notes:** Accent color is teal (#2A7B9B). Module background: var(--color-bg-warm) for alternating rhythm. Use actor colors: Agent=teal (--color-actor-2), Arena=amber (--color-actor-4), Experiment=plum (--color-actor-3), Backend=forest (--color-actor-5).
