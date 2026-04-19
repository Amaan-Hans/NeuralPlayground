# Module 5: Real Rat Data

### Teaching Arc
- **Metaphor:** Think of the Experiment classes as a cold case detective file. Scientists wired up real rats with electrodes, let them explore arenas, and recorded exactly which neuron fired at exactly which location. Those recordings are now a permanent historical record — the ground truth that any new theory must explain. NeuralPlayground downloads, caches, and provides access to those files.
- **Opening hook:** "Three real experiments from the 2000s changed neuroscience forever. The data from those experiments — electrode recordings from rats in actual arenas — is bundled into this codebase. Here is how the code finds, downloads, and uses it."
- **Key insight:** The Experiment classes provide two things: (1) real animal position trajectories (so your simulated agent can follow the same path a real rat took), and (2) neural spike data (so you can compare your model's firing patterns to actual neurons). This comparison is the whole point of the framework.
- **"Why should I care?":** When you add a new published dataset to this framework, you need to understand this pattern: inherit from Experiment, implement data loading from a remote GIN repository using pooch. Knowing this saves you an hour of guessing.

### Code Snippets (pre-extracted)

**Hafting2008Data class header — experiments/hafting_2008_data.py (lines 21-60):**
```python
class Hafting2008Data(Experiment):
    """Data class for Hafting et al. 2008.
    The data can be obtained from https://archive.norstore.no/
    This class only considers animal raw trajectories and neural recordings.

    Attributes
    ----------
    data_path: str
        if None, fetch the data from the NeuralPlayground data repository,
        else load data from given path
    arena_limits: ndarray (2,2)
        limits of the arena in the experiment
    data_per_animal: dict
        dictionary with all the data from the experiment
    recording_list: Pandas dataframe
        List of available data
    position: ndarray (n_samples, 2)
        array with the x and y position throughout recording
    head_direction: ndarray (n_samples, 2)
        array with the x and y head direction
    """
```

**datasets.py fetch function — neuralplayground/datasets.py (first function):**
```python
from neuralplayground.datasets import fetch_data_path
data_path = fetch_data_path("hafting_2008", subset=True)
```

**Using an experimental arena — typical usage pattern from codebase examples:**
```python
from neuralplayground.agents import Stachenfeld2018
from neuralplayground.arenas import Sargolini2006

env = Sargolini2006(use_behavioral_data=True)
agent = Stachenfeld2018()

obs, state = env.reset()
for step in range(env.total_number_of_steps):
    action = agent.act(obs)
    obs, state, reward = env.step(action)
    agent.update()
```

### Interactive Elements

- [x] **Code↔English translation** — use the experimental arena usage pattern. Line-by-line:
  - `env = Sargolini2006(use_behavioral_data=True)` → "Load a virtual replica of the Sargolini lab experiment, AND load the actual rat trajectory data from disk (or download it if not cached)"
  - `agent = Stachenfeld2018()` → "Create the brain model that will navigate this arena"
  - `obs, state = env.reset()` → "Place the agent at the same starting position a real rat used in the experiment"
  - `for step in range(env.total_number_of_steps):` → "Loop exactly as many steps as the real rat took during the experiment"
  - `action = agent.act(obs)` → "Ask the model where to go next"
  - `env.step(action)` → "Move — but since use_behavioral_data=True, the arena may override this with the real rat trajectory"
  - `agent.update()` → "Let the model learn from this position"
- [x] **Data flow animation** — 4 actors: GIN Repository, Pooch Cache, Experiment, Arena. Steps:
  1. User creates Hafting2008Data object
  2. Experiment checks: is the data already in ~/.NeuralPlayground/data/?
  3. If not: Pooch downloads from GIN remote repository
  4. If yes: Pooch returns cached local path
  5. Experiment loads .mat files, parses positions and spike times
  6. Arena loads Experiment data — uses real trajectory for simulation
  7. Comparison: model rate map vs real neural rate map
- [x] **Quiz** — 3 questions:
  1. "You have a new published dataset of rat recordings in a T-maze. What is the minimum you need to implement to add it to NeuralPlayground?" (Answer: a new class inheriting from Experiment, with data loading logic and a _find_data_path() method that uses pooch to fetch/cache the files)
  2. "A colleague runs an experiment with use_behavioral_data=False. What is different about their simulation compared to yours (use_behavioral_data=True)?" (Answer: with False, the agent navigates randomly; with True, the agent follows the exact same path a real rat took — making the comparison more meaningful)
  3. Debugging: "You run the code and get a download error on the first run, but it works on the second run. What is happening?" (Answer: Pooch is downloading and caching the data files — the first run triggers the download which may be slow/interrupted; after caching, it reads locally)
- [x] **Glossary tooltips** — GIN repository, Pooch, caching, spike train, rate map, autocorrelation, electrode, neural recording, .mat file (MATLAB), behavioral data

### Reference Files to Read
- `references/interactive-elements.md` → "Code ↔ English Translation Blocks", "Message Flow / Data Flow Animation", "Multiple-Choice Quizzes", "Glossary Tooltips", "Callout Boxes"
- `references/content-philosophy.md` → always
- `references/gotchas.md` → always

### Connections
- **Previous module:** "How Neurons Learn Space" — showed how the brain models learn; this module shows what they are compared against
- **Next module:** "Science at Scale" — shows SimulationManager for running many model-data comparisons systematically
- **Tone/style notes:** Accent color is teal (#2A7B9B). Module background: var(--color-bg). Include a callout about the landmark nature of these experiments — Hafting2008, Sargolini2006, and Wernle2018 are cited thousands of times. The data is real history.
