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
            room_depth: float
                room depth specified by the environment
            state_density: float
                density of agent states (should be proportional to the step-size)
            use_reward: bool
                If True, gate Hebbian updates by ReLU(TD error). Default False.
            reward_location: list [x, y]
                Coordinates of the reward site. Default [3.0, 3.0].
            td_alpha: float
                TD learning rate for V(s). Default 0.1.
            td_gamma: float
                Discount factor for TD updates. Default 0.9.
            n_pretrain_episodes: int
                Episodes of free exploration before reward gating is applied.
                Default 0 (gating active from episode 1).

        """
        super().__init__()
        self.mod_kwargs = mod_kwargs.copy()
        params = mod_kwargs["params"]
        self.room_widths = mod_kwargs["room_widths"]
        self.room_depths = mod_kwargs["room_depths"]
        self.state_densities = mod_kwargs["state_densities"]

        self.pars = copy.deepcopy(params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tem = model.Model(self.pars, self.device)
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

        # --- Reward / TD parameters ---
        self.use_reward = mod_kwargs.get("use_reward", False)
        self.reward_location = mod_kwargs.get("reward_location", [3.0, 3.0])
        self.td_alpha = mod_kwargs.get("td_alpha", 0.1)
        self.td_gamma = mod_kwargs.get("td_gamma", 0.9)
        self.n_pretrain_episodes = mod_kwargs.get("n_pretrain_episodes", 0)
        self.reward_state_ids = self._compute_reward_state_ids()
        self.episode_count = 0

        self.reset()

    def reset(self):
        """Initialise model and associated variables for training, set
        n_walk=-1 initially to account for the lack of actions at
        initialisation.
        """
        self.tem = model.Model(self.pars, self.device)
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
        # TD state: one V table per environment, reset on full agent reset
        if self.use_reward:
            self.V = [np.zeros(self.n_states[i]) for i in range(self.batch_size)]
        self.td_errors = []   # per-step list of (batch_size,) TD error arrays
        self.step_log = []    # per-step dicts for offline analysis

    def _compute_reward_state_ids(self):
        """Find the state index nearest to reward_location for each environment."""
        ids = []
        for i in range(self.batch_size):
            rw = self.room_widths[i]
            rd = self.room_depths[i]
            sd = self.state_densities[i]
            res_w = int(sd * rw)
            res_d = int(sd * rd)
            x_arr = np.linspace(-rw / 2 + 0.5 / sd, rw / 2 - 0.5 / sd, num=res_w)
            y_arr = np.linspace(-rd / 2 + 0.5 / sd, rd / 2 - 0.5 / sd, num=res_d)
            xy = np.stack(np.meshgrid(x_arr, y_arr), axis=-1)
            diff = (xy - np.array(self.reward_location, dtype=float)) ** 2
            ids.append(int(np.argmin(np.sum(diff, axis=-1))))
        return ids

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
                # Compute per-environment TD errors using the transition s→s'
                if self.use_reward:
                    td_errors_step = self._compute_and_update_td(
                        self.prev_observations, observations
                    )
                    self.td_errors.append(td_errors_step)
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

    def _compute_and_update_td(self, prev_obs, curr_obs):
        """Compute TD error for each environment and update V tables.

        Parameters
        ----------
        prev_obs : list of [state_id, object, pos]  length=batch_size   (s)
        curr_obs : list of [state_id, object, pos]  length=batch_size   (s')

        Returns
        -------
        td_scales : np.ndarray shape (batch_size,)
            ReLU(delta) for each environment, used to gate the Hebbian update.
        """
        td_scales = np.zeros(self.batch_size, dtype=np.float32)
        for i in range(self.batch_size):
            s = prev_obs[i][0]
            s_prime = curr_obs[i][0]
            # Reward is delivered on arrival at the reward state
            r = 1.0 if s_prime == self.reward_state_ids[i] else 0.0
            delta = r + self.td_gamma * self.V[i][s_prime] - self.V[i][s]
            self.V[i][s] += self.td_alpha * delta
            scale = float(np.maximum(0.0, delta))
            td_scales[i] = scale
            self.step_log.append(
                {
                    "episode": self.episode_count,
                    "env": i,
                    "s": s,
                    "s_prime": s_prime,
                    "reward": r,
                    "delta": delta,
                    "hebbian_scale": scale,
                    "V_s": self.V[i][s],
                }
            )
        return td_scales

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

        # TD gating is suppressed during the pretrain phase so TEM can build
        # stable structural representations before reward modulation begins.
        gating_active = (
            self.use_reward and self.episode_count >= self.n_pretrain_episodes
        )

        # td_history is a list of (batch_size,) ReLU(delta) arrays, one per
        # rollout step.  None when gating is off — the model receives a plain
        # 3-element step tuple and behaves identically to the original TEM.
        if gating_active and len(self.td_errors) >= self.pars["n_rollout"]:
            td_history = self.td_errors[-self.pars["n_rollout"]:]
        else:
            td_history = None

        # Collect all information in walk variable.
        # When td_history is not None a 4th element (td_scale tensor) is appended
        # to each step; model.forward() unpacks and passes it to hebbian().
        model_input = []
        obs_array = np.reshape(observations, (20, 16, 45))
        act_array = np.reshape(action_values, (20, 16))
        for i in range(self.pars["n_rollout"]):
            step = [
                locations[i],
                torch.from_numpy(obs_array[i]).type(torch.float32).to(self.device),
                act_array[i].tolist(),
            ]
            if td_history is not None:
                # Shape (batch_size,) — hebbian() reshapes to (batch_size,1,1)
                step.append(
                    torch.tensor(td_history[i], dtype=torch.float32).to(self.device)
                )
            model_input.append(step)

        self.final_model_input = model_input
        self.episode_count += 1
        # Trim the td_errors buffer to avoid unbounded growth.
        if self.use_reward:
            self.td_errors = self.td_errors[self.pars["n_rollout"]:]

        forward = self.tem(model_input, self.prev_iter)

        # Accumulate loss from forward pass
        loss = torch.tensor(0.0, device=self.device)
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
                        loss_weights.to(self.device) * torch.stack([i[env_i] for i in step.L])
                    )
                else:
                    env_visited[step.g[env_i]["id"]] = True
            step_loss = (
                torch.tensor(0, device=self.device)
                if not step_loss
                else torch.mean(torch.stack(step_loss, dim=0), dim=0)
            )
            # Save all separate components of loss for monitoring
            plot_loss = plot_loss + step_loss.detach().cpu().numpy()
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
        # Move model to device (GPU if available)
        self.tem = self.tem.to(self.device)
        print(f"TEM using device: {self.device}")
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
            "../../neuralplayground/agents/whittington_2020_extras/whittington_2020_model.py",
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
