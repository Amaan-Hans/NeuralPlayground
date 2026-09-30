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
                      eval_fn=None, eval_interval: int = 1000, eval_save_path: str = None,
                      rotate_environments: bool = False):
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
    rotate_environments : bool
        If True, reproduces the original torch_tem training regime
        (jbakermans/torch_tem's run.py): each of the batch's environments
        independently walks a variable-length "episode" (sampled between
        ``params["walk_it_min"]`` and a shrinking ``walk_it_max``/
        ``walk_it_window`` window - the exact schedule
        ``parameter_iteration`` already computes for other purposes), and
        the moment that walk runs out, THAT ONE environment slot is
        replaced with a freshly randomized instance of the same
        size/config (new object layout, same graph) - the other slots are
        completely unaffected and keep training on whatever they currently
        have. This reuses Model.init_walks()'s existing per-slot "new walk"
        reset (triggered by ``prev_iter[0].a[env_i] is None``, mirroring
        torch_tem's own ``prev_iter[0].a[env_i] = None``) rather than
        needing any agent/model changes. Default False, which reproduces
        this codebase's previous behaviour exactly (env.reset() once, fixed
        for the whole run) - existing analyses that assume env 0's identity
        stays fixed across training checkpoints depend on this default.

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

    if rotate_environments:
        # Lazy import: keeps this generic training-loop module from
        # depending on a specific agent's parameter schedule unless rotation
        # is actually requested.
        from neuralplayground.agents.whittington_2020_extras import (
            whittington_2020_parameters as _tem_parameters,
        )
        n_envs = env.batch_size
        walk_it_min = params["walk_it_min"]
        walk_it_max = params["walk_it_max"]
        walk_it_window = params["walk_it_window"]
        # Initial walk lengths sampled from the FULL window, mirroring
        # torch_tem run.py's very first walk generation before any
        # parameter_iteration() call.
        steps_until_switch = [
            int(np.random.randint(walk_it_min, walk_it_max)) for _ in range(n_envs)
        ]

    for i in range(n_episode):
        # Collect n_rollout steps, then do one gradient update.
        while agent.n_walk < params["n_rollout"]:
            actions = agent.batch_act(obs)
            obs, state, reward = env.step(actions, normalize_step=True)
        agent.update()

        if rotate_environments:
            _, _, _, _, walk_length_center, _ = _tem_parameters.parameter_iteration(
                agent.iter, params
            )
            for j in range(n_envs):
                steps_until_switch[j] -= 1
                if steps_until_switch[j] <= 0:
                    new_env = env.env_class(**env.batch_arg_env_params[j])
                    new_obs, _ = new_env.reset(
                        random_state=random_start, custom_state=None if random_start else [0, 0]
                    )
                    env.environments[j] = new_env
                    obs[j] = new_obs
                    agent.visited[j] = [False for _ in range(agent.n_states[j])]
                    agent.held_landmark[j] = None
                    # Mark the timestamp of this rotation so eval code can
                    # truncate its trailing window to never reach back into
                    # slot j's previous (different) environment layout.
                    agent.last_rotation_step[j] = len(agent.obs_history)
                    if agent.use_reward and agent.td is not None:
                        # V[j] is keyed by landmark id (0..n_landmarks-1), which
                        # is only a meaningful proxy for "context/place" within
                        # THIS environment instance's own landmark layout - a
                        # fresh rotation reshuffles which physical state each
                        # landmark id sits at, so carrying the old values over
                        # would inject stale, now-arbitrary value beliefs into
                        # the sensory stream until TD overwrites them again.
                        agent.td.reset_env(j)
                    # Signal the model: slot j's next chunk starts a brand-new
                    # walk - Model.init_walks() (called at the top of the next
                    # forward()) resets M[j], g_inf[j], x_inf[j] to fresh
                    # initial values when it sees a None action here.
                    agent.prev_iter[0].a[j] = None
                    low = max(1, int(round(walk_length_center - walk_it_window * 0.5)))
                    high = max(low + 1, int(round(walk_length_center + walk_it_window * 0.5)))
                    steps_until_switch[j] = int(np.random.randint(low, high))

        # Periodic evaluation: save plots and raw arrays every eval_interval
        # episodes, plus always at episode 1 (i == 0) so training-progress
        # plots have a checkpoint at the very start, not just from
        # eval_interval onward.
        if eval_fn is not None and ((i + 1) % eval_interval == 0 or i == 0):
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
