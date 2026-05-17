import random

import numpy as np

from neuralplayground.agents import AgentCore
from neuralplayground.arenas import Environment


def default_training_loop(agent: AgentCore, env: Environment, n_steps: int):
    """Default training loop for agents and environments that use a step-based
    update.

    Parameters
    ----------
    agent : AgentCore
        Agent to be trained.
    env : Environment
        Environment in which the agent is trained.
    n_steps : int
        Number of steps to train the agent for.

    Returns
    -------
    agent : AgentCore
        Trained agent.
    env : Environment
        Environment in which the agent was trained.
    dict_training : dict
        Dictionary containing the training history from the training loop and update
        method.

    """
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
    dict_training = process_training_hist(training_hist)
    return agent, env, dict_training


def episode_based_training_loop(
    agent: AgentCore, env: Environment, t_episode: int, n_episode: int
):
    """Training loop for agents and environments that use an episode-based
    update.

    Parameters
    ----------
    agent : AgentCore
        Agent to be trained.
    env : Environment
        Environment in which the agent is trained.
    t_episode : int
        Number of steps per episode.
    n_episode : int
        Number of episodes to train the agent for.

    Returns
    -------
    agent : AgentCore
        Trained agent.
    env : Environment
        Environment in which the agent was trained.
    dict_training : dict
        Dictionary containing the training history from the training loop and update
        method.

    """
    obs, state = env.reset()
    obs = obs[:2]
    training_hist = []
    for i in range(n_episode):
        for j in range(t_episode):
            action = agent.act(obs)
            update_output = agent.update()
            training_hist.append(update_output)
            obs, state, reward = env.step(action)
            obs = obs[:2]
    dict_training = process_training_hist(training_hist)
    return agent, env, dict_training


def tem_training_loop(agent: AgentCore, env: Environment, n_episode: int, params: dict,
                      trajectory_seed: int = None, random_start: bool = False,
                      eval_fn=None, eval_interval: int = 1000, eval_save_path: str = None):
    """Training loop for agents and environments that use a TEM-based update.

    Parameters
    ----------
    agent : AgentCore
        Agent to be trained.
    env : Environment
        Environment in which the agent is trained.
    n_episode : int
        Number of episodes (outer loop iterations) to train for.
    params : dict
        Dictionary of TEM model parameters, e.g. ``params["n_rollout"]``
        controls how many walk steps are collected before each update.
        Optional keys:
          ``trajectory_seed`` (int): seed np.random before trajectory begins
          so reward and no-reward runs follow identical paths.
          ``random_start`` (bool): if False (default), agents start at [0,0].

    Returns
    -------
    agent : AgentCore
        Trained agent.
    env : Environment
        Environment in which the agent was trained.
    training_dict : list
        List containing the agent kwargs, environment kwargs, and TEM
        hyperparameters recorded at the end of training.

    """
    training_dict = [agent.mod_kwargs, env.env_kwargs, agent.tem.hyper]

    # Seed both RNGs before env.reset() so object layouts (Python random) and
    # action sequences (numpy) are identical across the baseline and reward runs.
    if trajectory_seed is not None:
        random.seed(trajectory_seed)
        np.random.seed(trajectory_seed)

    # Fixed start position [0,0] keeps both conditions comparable; random_start
    # can be re-enabled for standard TEM training without the reward experiment.
    obs, state = env.reset(random_state=random_start, custom_state=None if random_start else [0, 0])
    for i in range(n_episode):
        # Collect n_rollout steps, then do one gradient update.
        while agent.n_walk < params["n_rollout"]:
            actions = agent.batch_act(obs)
            obs, state, reward = env.step(actions, normalize_step=True)
        agent.update()
        # Periodic evaluation: save plots and raw arrays every eval_interval episodes.
        if eval_fn is not None and (i + 1) % eval_interval == 0:
            eval_fn(agent, env, i + 1, eval_save_path)
    return agent, env, training_dict


def process_training_hist(training_hist):
    """Process the training history from the training loop and update method.

    Parameters
    ----------
    training_hist : list
        List of dictionaries containing the training history from the training loop and
        update method.

    Returns
    -------
    dict_training : dict
        Dictionary containing the one list per key in the training_hist. The list
        contains the values for
        that key for each step in the training loop.

    """
    dict_training = {}
    if training_hist[0] is None:
        dict_training = None
    else:
        for key in training_hist[0].keys():
            dict_training[key] = []
        for i in range(len(training_hist)):
            for key in training_hist[i].keys():
                dict_training[key].append(training_hist[i][key])
    return dict_training
