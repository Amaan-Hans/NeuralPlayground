# Module 6: Science at Scale

### Teaching Arc
- **Metaphor:** Running one simulation is like doing one experiment in a lab. Good science requires dozens of experiments with different conditions — varying the parameters, running each setup multiple times to account for randomness, comparing all results systematically. SimulationManager is the lab notebook that orchestrates all of that automatically.
- **Opening hook:** "A single simulation tells you almost nothing scientifically. You need to run Weber2018 vs Stachenfeld2018, in multiple arenas, with 5 random seeds each, and compare their grid scores. Without SimulationManager you would write 50 loops by hand. With it: three lines of code."
- **Key insight:** SimulationManager handles the combinatorial explosion of experiments: N agents × M arenas × K random seeds = N×M×K simulations, all run automatically, logged, and comparable. It is the difference between one-off exploration and reproducible science.
- **"Why should I care?":** When you ask AI to run a systematic comparison of two brain models, you can now say "use SimulationManager with runs_per_sim=5" instead of asking it to write 10 nested loops. Precise vocabulary → precise output.

### Code Snippets (pre-extracted)

**SimulationManager usage — typical pattern from default_simulation.py / examples:**
```python
from neuralplayground.backend import SimulationManager
from neuralplayground.backend.default_simulation import weber_in_2d, stachenfeld_in_2d

manager = SimulationManager(
    simulation_list=[weber_in_2d, stachenfeld_in_2d],
    runs_per_sim=5,
    manager_id="my_comparison"
)
manager.generate_sim_paths()
manager.run_all()
manager.check_run_status()
```

**SimulationManager class header — backend/simulation_manager.py (lines 15-80):**
```python
class SimulationManager(object):
    """Class to manage the runs of multiple combinations of agents,
    environments, parameters and training loops.

    Attributes
    ----------
    simulation_list: list of SingleSim objects
        List of SingleSim objects to run
    runs_per_sim: int
        Number of runs per simulation
    manager_id: str
        ID of the simulation manager

    Methods
    -------
    generate_sim_paths()
        Generate the paths for the simulations
    run_all()
        Run all the simulations in the list, runs_per_sim times
    check_run_status()
        Prints the status of the simulations
    """
```

**default_training_loop — backend/training_loops.py (lines 5-41):**
```python
def default_training_loop(agent: AgentCore, env: Environment, n_steps: int):
    obs, state = env.reset()
    training_hist = []
    obs = obs[:2]
    for j in range(round(n_steps)):
        action = agent.act(obs)
        obs, state, reward = env.step(action)
        update_output = agent.update()
        training_hist.append(update_output)
        obs = obs[:2]
    dict_training = process_training_hist(training_hist)
    return agent, env, dict_training
```

### Interactive Elements

- [x] **Code↔English translation** — use the SimulationManager usage pattern. Line-by-line:
  - `from ...default_simulation import weber_in_2d, stachenfeld_in_2d` → "Load two pre-configured simulation blueprints — these bundle together an agent class, an arena class, a training loop, and all parameters"
  - `SimulationManager(simulation_list=[...], runs_per_sim=5, manager_id=...)` → "Set up a manager to run each simulation 5 times (to average out randomness from different starting positions)"
  - `manager.generate_sim_paths()` → "Create the folder structure on disk where results will be saved"
  - `manager.run_all()` → "Execute all 10 simulations (2 configs × 5 runs each) — handles errors gracefully and logs failures"
  - `manager.check_run_status()` → "Print a status report: how many completed successfully, how many failed, and why"
- [x] **Group chat animation** — actors: SimulationManager (teal), Weber (amber), Stachenfeld (plum), Filesystem (forest). 
  - SimulationManager: "Starting run 1/5 for Weber2018..."
  - Weber: "Initializing 4,900 excitatory neurons and 900 inhibitory neurons."
  - SimulationManager: "Training loop running: 20,000 steps in Simple2D arena."
  - Weber: "Step 20,000 complete. Grid score: 0.73. Saving to disk."
  - Filesystem: "Saved: my_comparison/weber_in_2d/run_1/agent.pkl"
  - SimulationManager: "Run 1 complete. Starting run 1/5 for Stachenfeld2018..."
  - Stachenfeld: "Building successor representation matrix..."
  - SimulationManager: "All 10 runs complete. Weber avg score: 0.71. Stachenfeld avg score: 0.58."
- [x] **Quiz** — 3 scenario questions (this module wraps the whole course):
  1. "You add a new brain model called MyModel2024. What do you need to do to include it in a SimulationManager comparison?" (Answer: Create a SingleSim object specifying MyModel2024 as the agent_class, plus an arena class and training loop — then add it to simulation_list)
  2. "After running manager.run_all(), some simulations show status=FAILED. What should you check first?" (Answer: manager.show_logs(simulation_index, log_type='error') — SimulationManager logs all errors and stack traces automatically)
  3. Full-course trace: "Walk through the complete path from 'create a new Agent subclass' to 'compare its grid score to real rat data'" (Answer: 1) write Agent subclass with act() and update(); 2) create SingleSim config; 3) add to SimulationManager; 4) run_all(); 5) extract rate maps from trained agents; 6) use GridScorer.score() against Experiment data)
- [x] **Glossary tooltips** — random seed, reproducibility, grid score, spatial autocorrelation, rate map, pickle, serialization, stack trace, log file, benchmark

### Reference Files to Read
- `references/interactive-elements.md` → "Code ↔ English Translation Blocks", "Group Chat Animation", "Multiple-Choice Quizzes", "Glossary Tooltips", "Numbered Step Cards", "Callout Boxes"
- `references/content-philosophy.md` → always
- `references/gotchas.md` → always

### Connections
- **Previous module:** "Real Rat Data" — showed how ground truth data is loaded and used for comparison
- **Next module:** None (this is the last module) — end with a strong "what to do next" call to action
- **Tone/style notes:** Accent color is teal (#2A7B9B). Module background: var(--color-bg-warm). Close with a "What's next?" section listing 3 concrete next steps: (1) run the colab notebook, (2) implement a new Agent subclass, (3) add a new experimental dataset. This gives vibe coders something actionable to do immediately.
