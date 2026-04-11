import copy
import os
import pickle
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuralplayground.agents.whittington_2020_extras import (
    whittington_2020_analyse as analyse,
)
from neuralplayground.agents.whittington_2020_extras import (
    whittington_2020_model as model,
)
from neuralplayground.agents.whittington_2020_extras import (
    whittington_2020_parameters as parameters,
)
from neuralplayground.agents.whittington_2020_extras import (
    whittington_2020_utils as utils,
)

# Custom modules
from neuralplayground.plotting.plot_utils import make_plot_rate_map

from .agent_core import AgentCore

sys.path.append("../")


# =============================================================================
# TD Value Head — dopamine-modulated Hebbian learning
# =============================================================================
# Neuroscience background
# -----------------------
# Phasic dopamine from the ventral tegmental area (VTA) encodes a reward
# prediction error (RPE): delta = r + gamma*V(s') - V(s).  Positive RPEs
# (unexpected reward, or states that reliably predict reward) drive burst
# firing; negative RPEs drive dips below baseline.  In the hippocampus and
# entorhinal cortex, dopamine gates long-term potentiation: D1/D5 receptor
# activation during a burst converts a "tagged" synapse into lasting LTP.
# This means that place-cell memories formed just before a reward are
# consolidated more strongly than those far from reward, causing place fields
# to expand and shift backward along the approach trajectory — the temporal
# backward shift observed experimentally (Mehta et al. 2000; Stachenfeld et
# al. 2017).
#
# Implementation
# --------------
# TDValueHead learns a linear value function V(s) = w_v · p_inf[0](s) over
# the highest-frequency place cells (finest spatial resolution, 100-D).
# At each step it computes the TD(0) prediction error delta and returns an
# eta_scale: a multiplicative factor applied to the Hebbian learning rate
# eta inside model.hebbian().  Only positive deltas boost eta (dopamine
# burst); negative deltas are ignored here (dip effects are subtle and the
# simplest biologically-motivated version only requires the burst pathway).
#
# The value weights w_v are updated by a plain NumPy TD rule (no autograd)
# so they sit completely outside TEM's gradient graph.
class TDValueHead:
    """Linear critic over the highest-frequency place cells.

    Learns V(s) = w_v · p_inf[0](s) by TD(0).  Returns per-step, per-
    environment eta scale factors (1 + beta * max(delta, 0)) that the agent
    wrapper threads through forward() → iteration() → hebbian() to modulate
    Hebbian memory consolidation in a dopamine-like manner.

    Parameters
    ----------
    n_place_cells : int
        Dimensionality of p_inf[0] — the highest-frequency place cell
        module.  Equal to params["n_p"][0] (default 100).
    batch_size : int
        Number of parallel environments.
    gamma : float
        TD discount factor.  0.9 gives roughly a 10-step horizon, meaning
        states up to ~10 steps before reward accumulate meaningful value.
    lr : float
        Learning rate for the value weight update.
    beta : float
        Gain factor converting a positive TD error into an eta boost.
        eta_eff = eta_base * (1 + beta * max(delta, 0)).
        beta=3.0 means a delta of 1.0 quadruples the Hebbian rate at that
        step; beta=0 disables the modulation entirely.
    """

    def __init__(self, n_place_cells, batch_size, gamma=0.9, lr=0.01, beta=3.0):
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.beta = beta
        # One weight vector per environment so each arena learns its own
        # value function independently.
        self.w_v = np.zeros((batch_size, n_place_cells))

    def value(self, p_f0):
        """Compute V(s) for all environments.

        Parameters
        ----------
        p_f0 : np.ndarray, shape [batch_size, n_place_cells]
            Highest-frequency place cell activations for the current step,
            extracted from step.p_inf[0].detach().numpy().

        Returns
        -------
        V : np.ndarray, shape [batch_size]
        """
        return np.einsum("bi,bi->b", self.w_v, p_f0)

    def td_step(self, p_f0_t, p_f0_t1, rewards):
        """Compute TD error and update value weights.

        delta = r + gamma * V(s') - V(s)
        w_v  += lr * delta * p_f0(s)   (semi-gradient TD(0))

        Parameters
        ----------
        p_f0_t : np.ndarray, shape [batch_size, n_place_cells]
            Place cell activity at time t.
        p_f0_t1 : np.ndarray, shape [batch_size, n_place_cells]
            Place cell activity at time t+1.
        rewards : np.ndarray, shape [batch_size]
            Reward received on the transition t -> t+1.

        Returns
        -------
        delta : np.ndarray, shape [batch_size]
            The TD prediction error for each environment.
        """
        v_t = self.value(p_f0_t)
        v_t1 = self.value(p_f0_t1)
        delta = rewards + self.gamma * v_t1 - v_t
        # Semi-gradient update: treat V(s') as a constant (standard TD(0))
        self.w_v += self.lr * delta[:, np.newaxis] * p_f0_t
        return delta

    def eta_scale(self, delta):
        """Convert a TD error into a Hebbian rate multiplier.

        Only positive errors (dopamine bursts) boost consolidation.
        Negative errors are clamped to zero effect on eta so that the
        forgetting rate lambda handles the 'no-reward' signal separately
        (matching the biological observation that D1 receptor activation
        drives LTP, while reduced firing from baseline does not directly
        cause LTD in the same synapses).

        Parameters
        ----------
        delta : np.ndarray, shape [batch_size]

        Returns
        -------
        scale : np.ndarray, shape [batch_size]
            Multiplicative factor for eta_base.  Always >= 1.0.
        """
        return 1.0 + self.beta * np.maximum(delta, 0.0)


# =============================================================================


class Whittington2020(AgentCore):
    """Implementation of TEM 2020 by James C.R. Whittington, Timothy H. Muller,
    Shirley Mark, Guifen Chen, Caswell Barry, Neil Burgess, Timothy E.J.
    Behrens. The Tolman-Eichenbaum Machine: Unifying Space and Relational
    Memory through.

    Generalization in the Hippocampal Formation
    https://doi.org/10.1016/j.cell.2020.10.024.
    ----

    Attributes
    ----------
    mod_kwargs : dict
        Model parameters
        params: dict
            contains the majority of parameters used by the model and environment
        room_width: float
            room width specified by the environment
            (see examples/examples/whittington_2020_example.ipynb)
        room_depth: float
            room depth specified by the environment
            (see examples/examples/whittington_2020_example.ipynb)
        state_density: float
            density of agent states (should be proportional to the step-size)
        tem: class
            TEM model

    Methods
    -------
    reset(self):
        initialise model and associated variables for training
    def initialise(self):
        generate random distribution of objects and intialise optimiser,
        logger and relevant variables
    act(self, positions, policy_func):
        generates batch of random actions to be passed into the environment. If the
        returned positions are allowed, they are saved along with corresponding actions
    update(self):
        Perform forward pass of model and calculate losses and accuracies
    action_policy(self):
        random action policy that picks from [up, down, left right]
    discretise(self, step):
        convert (x,y) position into discrete location
    walk(self, positions):
        convert continuous positions into sequence of discrete locations
    make_observations(self, locations):
        observe what randomly distributed object is located at each position of a walk
    step_to_actions(self, actions):
        convert (x,y) action information into an integer value

    """

    def __init__(self, model_name: str = "TEM", **mod_kwargs):
        """Parameters
        ----------
        model_name : str
           Name of the specific instantiation of the ExcInhPlasticity class
        mod_kwargs : dict
            params: dict
                contains the majority of parameters used by the model and environment
            room_width: float
                room width specified by the environment
                (see examples/examples/whittington_2020_example.ipynb)
            room_depth: float
                room depth specified by the environment
                (see examples/examples/whittington_2020_example.ipynb)
            state_density: float
                density of agent states (should be proportional to the step-size)

        """
        super().__init__()
        self.mod_kwargs = mod_kwargs.copy()
        params = mod_kwargs["params"]
        self.room_widths = mod_kwargs["room_widths"]
        self.room_depths = mod_kwargs["room_depths"]
        self.state_densities = mod_kwargs["state_densities"]

        self.pars = copy.deepcopy(params)
        self.tem = model.Model(self.pars)
        self.batch_size = mod_kwargs["batch_size"]
        self.use_behavioural_data = mod_kwargs["use_behavioural_data"]
        self.n_envs_save = 4
        self.n_states = [
            int(self.room_widths[i] * self.room_depths[i] * self.state_densities[i])
            for i in range(self.batch_size)
        ]
        self.poss_actions = [[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]]
        self.n_actions = len(self.poss_actions)
        self.final_model_input = None
        self.g_rates, self.p_rates = None, None
        self.prev_observations = None

        # --- TD / dopamine state (Phase 2) ---
        # These are all None in Phase 1 and populated by activate_td_learning().
        # Keeping them as explicit attributes makes it easy to checkpoint and
        # inspect the value function alongside the TEM model weights.
        self.td_active = False          # switched on by activate_td_learning()
        self.value_head = None          # TDValueHead instance
        self.reward_ids = None          # list[int], reward state ID per env
        # eta scale factors from the *previous* rollout, applied to the
        # *current* rollout's Hebbian update (eligibility-trace-like delay).
        self.prev_eta_scales = None     # np.ndarray [n_rollout, batch_size]

        self.reset()

    def reset(self):
        """Initialise model and associated variables for training, set
        n_walk=-1 initially to account for the lack of actions at
        initialisation.
        """
        self.tem = model.Model(self.pars)
        self.initialise()
        self.n_walk = -1
        self.final_model_input = None
        self.obs_history = []
        self.walk_actions = []
        self.walk_action_values = []
        self.prev_action = None
        self.prev_observation = None
        self.prev_actions = [[None, None] for _ in range(self.batch_size)]
        self.prev_observations = [
            [-1, -1, [float("inf"), float("inf")]] for _ in range(self.batch_size)
        ]

    def act(self, observation, policy_func=None):
        """The base model executes one of four action (up-down-right-left) with
        equal probability. This is used to move on the rectangular environment
        states space (transmat). This is done for a single environment.

        Parameters
        ----------
        observation : array-like
            Observation from the environment (e.g. position) used to decide
            the next action.
        policy_func : callable, optional
            Inherited from AgentCore; not used in this model. To change the
            policy, override ``action_policy`` instead.

        Returns
        -------
        action : array (16,2)
            Action values (Direction of the agent step) in this case executes one of
            four action

        """
        new_action = self.action_policy()
        if observation[0] == self.prev_observation[0]:
            self.prev_action = new_action
        else:
            self.walk_actions.append(self.prev_action)
            self.obs_history.append(self.prev_observation)
            self.prev_action = new_action
            self.prev_observation = observation
            self.n_walk += 1

        return new_action

    def batch_act(self, observations, policy_func=None):
        """The base model executes one of four action (up-down-right-left) with
        equal probability. This is used to move on the rectangular environment
        states space (transmat). This is done for a batch of 16 environments.

        Parameters
        ----------
        observations: array (16,3/4)
            Observation from the environment class needed to choose the right action
            (here the state ID and position). If behavioural data is used, the
            observation includes head direction.

        Returns
        -------
        new_actions : array (16,2)
            Action values (direction of the agent step) in this case executes one of
            four action (up-down-right-left) with equal probability.

        """
        if self.use_behavioural_data:
            state_diffs = [
                observations[i][0] - self.prev_observations[i][0]
                for i in range(self.batch_size)
            ]
            new_actions = self.infer_action(state_diffs)
            self.walk_actions.append(new_actions)
            self.obs_history.append(self.prev_observations.copy())
            self.prev_observations = observations
            self.n_walk += 1

        elif not self.use_behavioural_data:
            locations = [env[0] for env in observations]
            all_allowed = True
            new_actions = []
            for i, loc in enumerate(locations):
                if loc == self.prev_observations[i][0] and self.prev_actions[i] != [
                    0,
                    0,
                ]:
                    all_allowed = False
                    break

            if all_allowed:
                self.walk_actions.append(self.prev_actions.copy())
                self.obs_history.append(self.prev_observations.copy())
                for batch in range(self.pars["batch_size"]):
                    new_actions.append(self.action_policy())
                self.prev_actions = new_actions
                self.prev_observations = observations
                self.n_walk += 1

            elif not all_allowed:
                for i, loc in enumerate(locations):
                    if loc == self.prev_observations[i][0]:
                        new_actions.append(self.action_policy())
                    else:
                        new_actions.append(self.prev_actions[i])
                self.prev_actions = new_actions

        return new_actions

    def update(self):
        """Compute forward pass through model, updating weights, calculating
        TEM variables and collecting losses / accuracies.
        """
        self.iter = int((len(self.obs_history) / 20) - 1)
        self.global_steps += 1
        history = self.obs_history[-self.pars["n_rollout"] :]
        locations = [
            [{"id": env_step[0], "shiny": None} for env_step in step]
            for step in history
        ]
        observations = [[env_step[1] for env_step in step] for step in history]
        actions = self.walk_actions[-self.pars["n_rollout"] :]
        self.n_walk = 0
        # Convert action vectors to action values
        action_values = self.step_to_actions(actions)
        self.walk_action_values.append(action_values)
        # Get start time for function timing
        time.time()
        # Get updated parameters for this backprop iteration
        (
            self.eta_new,
            self.lambda_new,
            self.p2g_scale_offset,
            self.lr,
            self.walk_length_center,
            loss_weights,
        ) = parameters.parameter_iteration(self.iter, self.pars)
        # Update eta and lambda
        self.tem.hyper["eta"] = self.eta_new
        self.tem.hyper["lambda"] = self.lambda_new
        # Update scaling of offset for variance of inferred grounded position
        self.tem.hyper["p2g_scale_offset"] = self.p2g_scale_offset
        # Update learning rate (the neater torch-way of doing this would be a scheduler,
        # but this is quick and easy)
        for param_group in self.adam.param_groups:
            param_group["lr"] = self.lr

        # Collect all information in walk variable
        model_input = [
            [
                locations[i],
                torch.from_numpy(np.reshape(observations, (20, 16, 45))[i]).type(
                    torch.float32
                ),
                np.reshape(action_values, (20, 16))[i].tolist(),
            ]
            for i in range(self.pars["n_rollout"])
        ]
        self.final_model_input = model_input

        # --- TD dopamine modulation: pass eta_scales from previous rollout ---
        # During Phase 1 (td_active=False) prev_eta_scales is None, so forward()
        # behaves exactly as the original model — no code path changes.
        #
        # During Phase 2 (td_active=True) we pass the eta scale factors computed
        # from the *previous* rollout's TD errors.  Using the previous rollout (not
        # the current one) avoids a chicken-and-egg problem (we need p_inf to
        # compute delta, but we need delta to set eta before getting p_inf).  It
        # also matches the eligibility-trace biology: activity tags the synapse
        # *during* the step, and the delayed dopamine burst consolidates it
        # *afterward*.  The one-rollout lag (~20 steps) is short relative to the
        # place field shift timescale.
        forward = self.tem(
            model_input,
            self.prev_iter,
            eta_scales=self.prev_eta_scales if self.td_active else None,
        )

        # After the forward pass, compute new TD errors and eta scales from this
        # rollout so they are ready for the next one.
        if self.td_active:
            self.prev_eta_scales = self._compute_td_scales(forward)

        # Accumulate loss from forward pass
        loss = torch.tensor(0.0)
        # Make vector for plotting losses
        plot_loss = 0
        # Collect all losses / variables
        for ind, step in enumerate(forward):
            # Make list of losses included in this step
            step_loss = []
            # Only include loss for locations that have been visited before
            for env_i, env_visited in enumerate(self.visited):
                if env_visited[step.g[env_i]["id"]]:
                    step_loss.append(
                        loss_weights * torch.stack([i[env_i] for i in step.L])
                    )
                else:
                    env_visited[step.g[env_i]["id"]] = True
            step_loss = (
                torch.tensor(0)
                if not step_loss
                else torch.mean(torch.stack(step_loss, dim=0), dim=0)
            )
            # Save all separate components of loss for monitoring
            plot_loss = plot_loss + step_loss.detach().numpy()
            # And sum all components, then add them to total loss of this step
            loss = loss + torch.sum(step_loss)

        # Reset gradients
        self.adam.zero_grad()
        # Do backward pass to calculate gradients with respect to total loss of this
        # chunk
        loss.backward(retain_graph=True)
        # Then do optimiser step to update parameters of model
        self.adam.step()
        # Update the previous iteration for the next chunk with the final step of
        # this chunk, removing all operation history
        self.prev_iter = [forward[-1].detach()]

        # Compute model accuracies
        acc_p, acc_g, acc_gt = np.mean(
            [[np.mean(a) for a in step.correct()] for step in forward], axis=0
        )
        acc_p, acc_g, acc_gt = [a * 100 for a in (acc_p, acc_g, acc_gt)]

    def initialise(self):
        """Generate random distribution of objects and intialise optimiser,
        logger and relevant variables.
        """
        # Create a logger to write log output to file
        current_dir = os.path.dirname(os.getcwd())
        run_path = os.path.join(current_dir, "agent_examples", "results_sim")
        run_path = os.path.normpath(run_path)
        self.logger = utils.make_logger(run_path)
        # Make an ADAM optimizer for TEM
        self.adam = torch.optim.Adam(self.tem.parameters(), lr=self.pars["lr_max"])
        # Initialise whether a state has been visited for each world
        self.visited = [
            [False for _ in range(self.n_states[env])]
            for env in range(self.pars["batch_size"])
        ]
        self.prev_iter = None

    def save_agent(self, save_path: str):
        """Save current state and information in general to re-instantiate the
        agent.

        Parameters
        ----------
        save_path: str
            Path to save the agent

        """
        pickle.dump(
            self.tem.state_dict(),
            open(os.path.join(save_path), "wb"),
            pickle.HIGHEST_PROTOCOL,
        )
        with open(os.path.join(os.path.dirname(save_path), "agent_hyper"), "wb") as fp:
            pickle.dump(self.tem.hyper, fp, pickle.HIGHEST_PROTOCOL)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        source_file_path = os.path.join(
            script_dir,
            """../../neuralplayground/agents/whittington_2020_extras/
            whittington_2020_model.py""",
        )
        destination_folder = os.path.join(os.path.dirname(save_path))
        os.makedirs(destination_folder, exist_ok=True)
        destination_file_path = os.path.join(
            destination_folder, "whittington_2020_model.py"
        )
        shutil.copy(source_file_path, destination_file_path)

    def save_files(self):
        """Save all python files in current directory to script directory."""
        curr_path = os.path.dirname(os.path.abspath(__file__))
        shutil.copy2(
            os.path.abspath(
                os.path.join(
                    os.getcwd(), os.path.abspath(os.path.join(curr_path, os.pardir))
                )
            )
            + "/agents/whittington_2020_extras/whittington_2020_model.py",
            os.path.join(self.script_path, "whittington_2020_model.py"),
        )
        shutil.copy2(
            os.path.abspath(
                os.path.join(
                    os.getcwd(), os.path.abspath(os.path.join(curr_path, os.pardir))
                )
            )
            + "/agents/whittington_2020_extras/whittington_2020_parameters.py",
            os.path.join(self.script_path, "whittington_2020_parameters.py"),
        )
        shutil.copy2(
            os.path.abspath(
                os.path.join(
                    os.getcwd(), os.path.abspath(os.path.join(curr_path, os.pardir))
                )
            )
            + "/agents/whittington_2020_extras/whittington_2020_analyse.py",
            os.path.join(self.script_path, "whittington_2020_analyse.py"),
        )
        shutil.copy2(
            os.path.abspath(
                os.path.join(
                    os.getcwd(), os.path.abspath(os.path.join(curr_path, os.pardir))
                )
            )
            + "/agents/whittington_2020_extras/whittington_2020_utils.py",
            os.path.join(self.script_path, "whittington_2020_utils.py"),
        )
        return

    # =========================================================================
    # TD / dopamine modulation — Phase 2 interface
    # =========================================================================

    def activate_td_learning(self, reward_ids, gamma=0.9, lr=0.01, beta=3.0):
        """Switch the agent from unsupervised TEM (Phase 1) to TD-modulated
        Hebbian learning (Phase 2).

        Call this *after* Phase 1 training has converged so that the grid/
        place cell representations are already meaningful before reward
        information is introduced.  This matches the biological situation
        where spatial representations are formed first during exploration,
        and dopaminergic modulation then sculpts them based on reward value.

        Parameters
        ----------
        reward_ids : list[int], length batch_size
            State ID of the reward location in each parallel environment.
            Use env.n_states[i] * fraction to place it at any position.
        gamma : float
            TD discount factor (default 0.9 → ~10 step horizon).
        lr : float
            Value function learning rate (default 0.01).
        beta : float
            Hebbian rate gain per unit of positive TD error.
            eta_eff = eta_base * (1 + beta * max(delta, 0)).
        """
        self.reward_ids = reward_ids
        self.value_head = TDValueHead(
            n_place_cells=self.pars["n_p"][0],  # highest-frequency module
            batch_size=self.batch_size,
            gamma=gamma,
            lr=lr,
            beta=beta,
        )
        self.prev_eta_scales = None  # will be computed after first Phase 2 rollout
        self.td_active = True

    def _get_reward_signals(self, forward):
        """Return a reward array for each (step, environment) pair.

        Reward = 1.0 when the agent is at the designated reward location,
        0.0 otherwise.  The random-walk policy means the agent visits the
        reward location by chance; we do not need to change the policy.

        Parameters
        ----------
        forward : list[Iteration]
            Output of self.tem(model_input, ...) — one Iteration per step.

        Returns
        -------
        rewards : np.ndarray, shape [n_steps, batch_size]
        """
        n_steps = len(forward)
        rewards = np.zeros((n_steps, self.batch_size))
        for step_i, step in enumerate(forward):
            for env_i in range(self.batch_size):
                if step.g[env_i]["id"] == self.reward_ids[env_i]:
                    rewards[step_i, env_i] = 1.0
        return rewards

    def _extract_place_cells(self, step):
        """Return the highest-frequency place cell activations as numpy array.

        Parameters
        ----------
        step : Iteration

        Returns
        -------
        p_f0 : np.ndarray, shape [batch_size, n_p[0]]
            Detached from the autograd graph so it can be used in NumPy TD
            updates without polluting TEM's gradient.
        """
        return step.p_inf[0].detach().numpy()

    def _compute_td_scales(self, forward):
        """Compute per-step, per-environment eta scale factors from TD errors.

        Iterates over consecutive pairs of steps in the rollout, computes
        TD(0) prediction errors using the linear value function, updates the
        value weights, then converts deltas into Hebbian rate multipliers.

        The resulting scale array is stored as self.prev_eta_scales and will
        be passed to forward() on the *next* rollout.  This one-rollout delay
        is the eligibility-trace analogue: the synapse is tagged by activity
        now, and the dopamine signal (arriving after the reward is processed)
        determines whether it is potentiated.

        Parameters
        ----------
        forward : list[Iteration], length n_rollout

        Returns
        -------
        eta_scales : np.ndarray, shape [n_rollout, batch_size]
            Values are 1.0 (no boost) everywhere except at steps where a
            positive TD error was observed, where they are > 1.0.
        """
        n_steps = len(forward)
        rewards = self._get_reward_signals(forward)
        # Default: no boost (scale = 1.0 everywhere)
        eta_scales = np.ones((n_steps, self.batch_size))

        for step_i in range(n_steps - 1):
            p_t = self._extract_place_cells(forward[step_i])
            p_t1 = self._extract_place_cells(forward[step_i + 1])
            r = rewards[step_i]  # reward on transition t -> t+1
            delta = self.value_head.td_step(p_t, p_t1, r)
            eta_scales[step_i] = self.value_head.eta_scale(delta)

        # Last step: no s' available, leave scale at 1.0
        return eta_scales

    # =========================================================================

    def action_policy(self):
        """Random action policy that selects an action to take from [stay, up,
        down, left, right]
        """
        arrow = self.poss_actions
        index = np.random.choice(len(arrow))
        action = arrow[index]
        return action

    def step_to_actions(self, actions):
        """Convert trajectory of (x,y) actions into integer values (i.e. from
        [[0,0],[0,-1],[1,0],[0,1],[-1,0]] to [0,1,2,3,4])

        Parameters
        ----------
            actions: (16,20,2)
                batch of 16 actions for each step in a walk of length 20

        Returns
        -------
            action_values: (16,20,1)
                batch of 16 action values for each step in walk of length 20

        """
        action_values = []
        # actions = np.reshape(actions, (pars['n_rollout'], pars['batch_size'], 2))
        for steps in actions:
            env_list = []
            for action in steps:
                env_list.append(self.poss_actions.index(list(action)))
            action_values.append(env_list)
        return action_values

    def infer_action(self, state_diffs):
        """Infers the action taken between state indices based on the
        difference between states.

        Parameters
        ----------
        state_diffs : list of int, length batch_size
            Difference between consecutive state indices for each environment
            in the batch.

        Returns
        -------
        actions : list of list of int, shape (batch_size, 2)
            Inferred [dx, dy] action vectors for each environment.

        """
        actions = []
        for i in range(self.batch_size):
            if state_diffs[i] == -self.room_widths[i]:
                actions.append([0, 1])
            elif state_diffs[i] == self.room_widths[i]:
                actions.append([0, -1])
            elif state_diffs[i] == -1:
                actions.append([-1, 0])
            elif state_diffs == 1:
                actions.append([1, 0])
            else:
                actions.append([0, 0])

        return actions

    def collect_final_trajectory(self):
        """Collect the final trajectory of the agent, including the locations,
        observations and actions taken.
        """
        final_model_input = []
        environments = [
            [],
            self.n_actions,
            self.n_states[0],
            len(self.obs_history[-1][0][1]),
        ]
        history = self.obs_history[-self.n_walk :]
        locations = [
            [{"id": env_step[0], "shiny": None} for env_step in step]
            for step in history
        ]
        observations = [[env_step[1] for env_step in step] for step in history]
        actions = self.walk_actions[-self.n_walk :]
        action_values = self.step_to_actions(actions)

        model_input = [
            [
                locations[i],
                torch.from_numpy(
                    np.reshape(observations, (self.n_walk, 16, 45))[i]
                ).type(torch.float32),
                np.reshape(action_values, (self.n_walk, 16))[i].tolist(),
            ]
            for i in range(self.n_walk)
        ]

        single_index = [[model_input[step][0][0]] for step in range(len(model_input))]
        single_obs = [
            torch.unsqueeze(model_input[step][1][0], dim=0)
            for step in range(len(model_input))
        ]
        single_action = [[model_input[step][2][0]] for step in range(len(model_input))]
        single_model_input = [
            [single_index[step], single_obs[step], single_action[step]]
            for step in range(len(model_input))
        ]
        final_model_input.extend(single_model_input)

        return final_model_input, history, environments

    def plot_run(self, tem, model_input, environments):
        with torch.no_grad():
            forward = tem(model_input, prev_iter=None)
        include_stay_still = False
        shiny_envs = [False, False, False, False]
        env_to_plot = 0
        (
            shiny_envs
            if shiny_envs[env_to_plot]
            else [not shiny_env for shiny_env in shiny_envs]
        )
        correct_model, correct_node, correct_edge = analyse.compare_to_agents(
            forward, tem, environments, include_stay_still=include_stay_still
        )
        analyse.zero_shot(
            forward, tem, environments, include_stay_still=include_stay_still
        )
        analyse.location_occupation(forward, tem, environments)
        self.g_rates, self.p_rates = analyse.rate_map(forward, tem, environments)
        from_acc, to_acc = analyse.location_accuracy(forward, tem, environments)
        return

    def plot_rate_map(
        self,
        rate_map_type=None,
        frequencies=["Theta", "Delta", "Beta", "Gamma", "High Gamma"],
        max_cells=30,
        num_cols=6,
    ):
        """Plot the TEM rate maps."""
        figs = []
        axes = []

        if rate_map_type == "g":
            rate_maps = self.g_rates
        elif rate_map_type == "p":
            rate_maps = self.p_rates
        if self.g_rates is None or self.p_rates is None:
            print("rate_maps must be of correct type")
            return
        for i in range(len(frequencies)):
            n_cells = rate_maps[0][i].shape[1]
            n_cells = min(n_cells, max_cells)
            # Number of subplots per row
            num_rows = np.ceil(n_cells / num_cols).astype(int)

            # Create the figure for the current frequency
            fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))
            fig.suptitle(f"{frequencies[i]} Rate Maps", fontsize=16)

            # Create the subplots for the current frequency
            for j in range(n_cells):
                if j >= n_cells:
                    break
                ax_row = j // num_cols
                ax_col = j % num_cols

                # Reshape the rate map into a matrix
                rate_map_mat = self.get_rate_map_matrix(rate_maps, i, j)

                # Plot the rate map in the corresponding subplot
                make_plot_rate_map(
                    rate_map_mat, axs[ax_row, ax_col], f"Cell {j + 1}", "", "", ""
                )

            # Hide unused subplots for the current frequency
            for j in range(n_cells, num_rows * num_cols):
                ax_row = j // num_cols
                ax_col = j % num_cols
                axs[ax_row, ax_col].axis("off")

            figs.append(fig)
            axes.append(axs)
        return figs, axes

    def get_rate_map_matrix(self, rate_maps, i, j):
        """Return a 2-D rate-map matrix for a single cell and environment.

        Parameters
        ----------
        rate_maps : list
            Nested list of rate maps as returned by the model, indexed
            ``[frequency][env_idx][cell_idx]``.
        i : int
            Environment index within ``rate_maps[0]``.
        j : int
            Cell index within the transposed rate-map array.

        Returns
        -------
        rate_map : np.ndarray of shape (room_width, room_depth)
            2-D rate-map for cell ``j`` in environment ``i``.

        """
        rate_map = np.asarray(rate_maps[0][i]).T[j]
        return np.reshape(rate_map, (self.room_widths[0], self.room_depths[0]))
