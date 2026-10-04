import os

import pandas as pd

from neuralplayground.agents.whittington_2020 import Whittington2020


def agent_with_optional_checkpoint(**kwargs):
    """Agent factory that optionally loads pretrained weights from the
    TEM_LOAD_CHECKPOINT env var (see whittington_2020_run.py) right after
    construction.

    Defined as a real, importable top-level function - not a closure
    defined inline in the run script - because SingleSim pickles
    agent_class by reference into params.dict. A function living in a
    script's own __main__ module can't be found when a different script
    (e.g. the post-hoc prediction-accuracy probe) later unpickles that
    checkpoint; a function imported from an actual package module can.
    """
    agent = Whittington2020(**kwargs)
    checkpoint_path = os.environ.get("TEM_LOAD_CHECKPOINT")
    if checkpoint_path:
        state_dict = pd.read_pickle(checkpoint_path)
        agent.tem.load_state_dict(state_dict)
        print(f"---> Loaded pretrained weights from {checkpoint_path}")
    return agent
