"""Periodic evaluation and plotting utilities for TEM experiments.

Called every eval_interval episodes from tem_training_loop.
Produces plots for environment 0 only:
  - trajectory.png
  - value_map.png          (reward condition only)
  - object_value_map.png   (reward condition only)
  - place_cells_<freq>.png (one per frequency module)
  - grid_cells_<freq>.png  (one per frequency module)

Folder layout:
  <eval_save_path>/plots/episode_<N>/<plot>.png
"""

import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # non-interactive: safe for long training runs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

FREQ_NAMES = ["Theta", "Delta", "Beta", "Gamma", "High_Gamma"]
EVAL_STEPS = 500   # history window used for rate maps


def run_eval(agent, env, episode: int, eval_save_path: str):
    """Entry point called from the training loop every eval_interval episodes.

    Parameters
    ----------
    agent : Whittington2020
    env   : BatchEnvironment
    episode : int  -- current episode number (1-indexed)
    eval_save_path : str  -- root save directory (e.g. results_sim/baseline)
    """
    ep_dir = os.path.join(eval_save_path, "plots", f"episode_{episode}")
    os.makedirs(ep_dir, exist_ok=True)

    n_hist = len(agent.obs_history)
    if n_hist == 0:
        return

    # Filter out the dummy placeholder rows (state_id == -1) that batch_act
    # inserts for env 0 on the very first step before any real observation.
    # held_landmark_history is appended in lockstep with obs_history (see
    # agent.batch_act), so the same index filtering keeps them aligned.
    real_indices = [i for i, step in enumerate(agent.obs_history) if step[0][0] != -1]
    real_history = [agent.obs_history[i] for i in real_indices]
    real_actions = agent.walk_actions[-len(real_history):]

    n_steps = min(EVAL_STEPS, len(real_history))
    if n_steps == 0:
        return
    history_slice = real_history[-n_steps:]
    held_indices = real_indices[-n_steps:]
    walk_slice = real_actions[-n_steps:]

    n_obs = len(history_slice[0][0][1])
    n_states = agent.n_states[0]
    room_w = int(agent.room_widths[0])
    room_d = int(agent.room_depths[0])

    # ── Build single-env (env 0) model input ──────────────────────────────────
    locations_seq = [[{"id": step[0][0], "shiny": None}] for step in history_slice]
    obs_seq = np.array([step[0][1] for step in history_slice], dtype=np.float32)

    # V(held landmark) reaches TEM via Model.inf_p's f_v bias, passed as a
    # separate model_input element (never concatenated onto the observation)
    # — mirrors agent._value_for_history used during training.
    v_seq = None
    if agent.use_reward and agent.td is not None:
        v_table = agent.td.V[0]
        v_max = float(np.max(v_table))
        v_seq = np.zeros(obs_seq.shape[0], dtype=np.float32)
        for i, idx in enumerate(held_indices):
            key = agent.held_landmark_history[idx][0]
            if key is None:
                continue
            v_t = float(v_table[key]) if 0 <= key < v_table.shape[0] else 0.0
            v_seq[i] = v_t / v_max if v_max > 0 else 0.0

    action_values = agent.step_to_actions(walk_slice)
    action_array = np.reshape(action_values, (n_steps, 16))[:, 0]

    model_input = [
        [
            locations_seq[i],
            torch.tensor(obs_seq[i : i + 1], dtype=torch.float32).to(agent.device),
            [int(action_array[i])],
        ]
        for i in range(n_steps)
    ]
    if v_seq is not None:
        for i in range(n_steps):
            model_input[i].append(None)  # td_scale slot (Hebbian gating) - unused
            model_input[i].append(
                torch.tensor([v_seq[i]], dtype=torch.float32).to(agent.device)
            )

    # ── Forward pass (no gradient, restore batch_size afterwards) ─────────────
    saved_batch_size = agent.tem.hyper.get("batch_size", 16)
    agent.tem.eval()
    with torch.no_grad():
        forward = agent.tem(model_input, prev_iter=None)
    agent.tem.train()
    agent.tem.hyper["batch_size"] = saved_batch_size

    # ── Compute rate maps ─────────────────────────────────────────────────────
    n_f = agent.pars["n_f"]
    n_p_list = agent.pars["n_p"]
    n_g_list = agent.pars["n_g"]

    p_accum = [[[] for _ in range(n_states)] for _ in range(n_f)]
    g_accum = [[[] for _ in range(n_states)] for _ in range(n_f)]

    for step in forward:
        loc = step.g[0]["id"]
        for f in range(n_f):
            p_accum[f][loc].append(step.p_inf[f][0].detach().cpu().numpy())
            g_accum[f][loc].append(step.g_inf[f][0].detach().cpu().numpy())

    def _average_second_half(accum, n_cells_list):
        rates = []
        for f in range(n_f):
            freq_rates = []
            for loc in range(n_states):
                visits = accum[f][loc]
                if not visits:
                    freq_rates.append(np.zeros(n_cells_list[f]))
                else:
                    half = len(visits) // 2
                    usable = visits[half:] if half < len(visits) else visits
                    freq_rates.append(np.mean(usable, axis=0))
            rates.append(np.stack(freq_rates, axis=0))  # (n_states, n_cells_f)
        return rates

    p_rates = _average_second_half(p_accum, n_p_list)
    g_rates = _average_second_half(g_accum, n_g_list)

    # ── Save raw arrays for post-hoc predictive-coding analysis ───────────────
    # p_rates: list of n_f arrays, each (n_states, n_cells_f)
    # Concatenate across frequencies -> (n_states, total_p_cells)
    p_all = np.concatenate(p_rates, axis=1)
    np.save(os.path.join(ep_dir, "p_rates.npy"), p_all)

    g_all = np.concatenate(g_rates, axis=1)
    np.save(os.path.join(ep_dir, "g_rates.npy"), g_all)

    if agent.use_reward and agent.td is not None:
        # V is keyed by landmark identity (agent.td.V[0] has shape
        # (n_landmarks,)), not by state. Each landmark occupies exactly one
        # state in this environment's layout, so project the table onto
        # those states for spatial plotting; every other (non-landmark)
        # state has no fixed value of its own (its "context" while passing
        # through is whatever landmark was last held, which is path-
        # dependent) and is left as NaN rather than implying a value of 0.
        object_layout = env.environments[0].objects  # (n_states, n_objects) one-hot
        object_ids_per_state = np.argmax(object_layout, axis=1)
        v_per_state = np.full(n_states, np.nan, dtype=np.float32)
        for sid in range(n_states):
            obj_id = int(object_ids_per_state[sid])
            if obj_id < agent.n_landmarks:
                v_per_state[sid] = agent.td.V[0][obj_id]
        np.save(os.path.join(ep_dir, "v_table.npy"), v_per_state)

    # ── 1. Trajectory ─────────────────────────────────────────────────────────
    positions = [step[0][2] for step in history_slice]
    xs = [float(p[0]) for p in positions]
    ys = [float(p[1]) for p in positions]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, color="steelblue", alpha=0.45, linewidth=0.8)
    ax.scatter(xs[0], ys[0], c="green", s=70, zorder=5, label="start")
    ax.scatter(xs[-1], ys[-1], c="red", s=70, zorder=5, label="end")
    if agent.use_reward:
        rx, ry = agent.reward_location
        ax.scatter([rx], [ry], c="gold", s=200, marker="*", zorder=6, label="reward")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Trajectory – env 0 – episode {episode}\n(last {n_steps} steps)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ep_dir, "trajectory.png"), dpi=150)
    plt.close(fig)

    # ── 2. Value map (reward condition only) ──────────────────────────────────
    # NaN (non-landmark states) renders as gray via cmap.set_bad, so the
    # n_landmarks coloured cells stand out directly against a blank background.
    if agent.use_reward and agent.td is not None and n_states == room_d * room_w:
        v_grid = np.reshape(v_per_state, (room_d, room_w))
        cmap = plt.get_cmap("hot").copy()
        cmap.set_bad(color="gray")

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(v_grid, origin="lower", cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label="V(landmark)")
        rx, ry = agent.reward_location
        ax.scatter([rx + room_w / 2 - 0.5], [ry + room_d / 2 - 0.5], c="cyan", s=120,
                   marker="*", zorder=6, label="reward")
        ax.legend(fontsize=7)
        ax.set_title(f"Landmark value V — env 0 – episode {episode}\n"
                     f"(gray = non-landmark state, no fixed value of its own)")
        ax.set_xlabel("x bin")
        ax.set_ylabel("y bin")
        fig.tight_layout()
        fig.savefig(os.path.join(ep_dir, "value_map.png"), dpi=150)
        plt.close(fig)

        # ── 2b. Object identity overlaid on landmark value ──────────────────────
        # Checks the thing landmark placement is meant to produce: landmarks
        # sampled closer to the reward (see DiscreteObjectEnvironment's
        # landmark_bias_scale) should end up with higher learned V than ones
        # placed farther away.
        object_layout = env.environments[0].objects  # (n_states, n_objects) one-hot
        object_ids_per_state = np.argmax(object_layout, axis=1)
        obj_grid = np.reshape(object_ids_per_state, (room_d, room_w))
        is_landmark_grid = np.reshape(object_ids_per_state < agent.n_landmarks, (room_d, room_w))

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(v_grid, origin="lower", cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label="V(landmark)")

        for row in range(room_d):
            for col in range(room_w):
                ax.text(
                    col, row, str(int(obj_grid[row, col])),
                    ha="center", va="center", fontsize=6, color="cyan",
                )
                if is_landmark_grid[row, col]:
                    ax.add_patch(
                        mpatches.Rectangle(
                            (col - 0.5, row - 0.5), 1, 1,
                            fill=False, edgecolor="lime", linewidth=2,
                        )
                    )
        ax.scatter([rx + room_w / 2 - 0.5], [ry + room_d / 2 - 0.5], c="cyan", s=120,
                   marker="*", zorder=6, label="reward")
        ax.legend(fontsize=7)

        ax.set_title(
            f"Object id (cyan text) + landmark V – env 0 – episode {episode}\n"
            f"lime boxes = the {agent.n_landmarks} landmark states (ids 0..{agent.n_landmarks - 1})"
        )
        ax.set_xlabel("x bin")
        ax.set_ylabel("y bin")
        fig.tight_layout()
        fig.savefig(os.path.join(ep_dir, "object_value_map.png"), dpi=150)
        plt.close(fig)

    # ── 3 & 4. Place cell and Grid cell rate maps ─────────────────────────────
    _save_rate_maps(p_rates, n_p_list, room_w, room_d, ep_dir, "place_cells", episode)
    _save_rate_maps(g_rates, n_g_list, room_w, room_d, ep_dir, "grid_cells", episode)

    print(f"  [eval ep {episode}] plots saved -> {ep_dir}", flush=True)


def _save_rate_maps(rates, n_cells_list, room_w, room_d, ep_dir, prefix, episode):
    """Save one figure per frequency module."""
    num_cols = 6
    for f_idx, (freq_name, freq_rates) in enumerate(zip(FREQ_NAMES, rates)):
        max_cells = min(30, n_cells_list[f_idx])
        if max_cells == 0:
            continue
        num_rows = int(np.ceil(max_cells / num_cols))

        fig, axs = plt.subplots(
            nrows=num_rows, ncols=num_cols,
            figsize=(15, max(3, num_rows * 2.5)),
        )
        # Normalise axs to always be 2-D for uniform indexing
        if num_rows == 1:
            axs = axs[np.newaxis, :]

        label = prefix.replace("_", " ").title()
        fig.suptitle(f"{label} – {freq_name} – episode {episode}", fontsize=12)

        for j in range(max_cells):
            row, col = j // num_cols, j % num_cols
            cell_map = freq_rates[:, j]          # (n_states,)
            cell_2d = np.reshape(cell_map, (room_d, room_w))
            axs[row, col].imshow(cell_2d, origin="lower", cmap="viridis", aspect="auto")
            axs[row, col].set_title(f"Cell {j + 1}", fontsize=7)
            axs[row, col].axis("off")

        for j in range(max_cells, num_rows * num_cols):
            axs[j // num_cols, j % num_cols].axis("off")

        fig.tight_layout()
        fname = f"{prefix}_{freq_name}.png"
        fig.savefig(os.path.join(ep_dir, fname), dpi=150)
        plt.close(fig)
