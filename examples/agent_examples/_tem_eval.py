"""Periodic evaluation and plotting utilities for TEM experiments.

run_eval() is called every eval_interval episodes from tem_training_loop.
Produces plots for environment 0 only:
  - trajectory.png
  - value_map.png          (reward condition only)
  - object_value_map.png   (reward condition only)
  - place_cells_<freq>.png (one per frequency module)
  - grid_cells_<freq>.png  (one per frequency module)

compute_multienv_rates() is NOT called from the training loop - it's a
small-window pure-computation core (used by run_eval_multienv() below, a
thin save-to-plots/ wrapper kept for any caller that wants that convention).

run_multienv_probe() is the long frozen-weight probe's own driver, used by
tem_probe_eval_multienv.py against an already-trained agent: it runs the
random walk itself (not just processes a pre-existing one), interleaved
chunk-by-chunk with the rate-map forward pass, discarding each chunk's raw
history immediately so memory stays flat regardless of walk length. Its
result is saved directly under results_sim/predictive_analysis/probe/, not
into plots/episode_<N>/, since that folder is scanned by several other
analyses (population activity, peak distance, grid scores, proximal cell
count) that should only ever see real training checkpoints.

Folder layout:
  <eval_save_path>/plots/episode_<N>/<plot>.png
"""

import os
import pickle
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # non-interactive: safe for long training runs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

FREQ_NAMES = ["Theta", "Delta", "Beta", "Gamma", "High_Gamma"]
EVAL_STEPS = 500   # history window used for rate maps
MIN_DISPLAY_AMPLITUDE = 0.02   # matches tem_predictive_analysis.py's MIN_AMPLITUDE convention


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

    # V(held landmark) reaches TEM by being written into the compressed code
    # x_c's dedicated trailing dimension inside Model.inference() (right after
    # f_c's argmax/lookup), passed here as a separate model_input element
    # (never concatenated onto the raw observation) — mirrors
    # agent._value_for_history used during training.
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

    # Visit count per state within this eval window - same for every f since
    # accum[f][loc] gets exactly one append per step regardless of frequency
    # (see the accumulation loop above), so f=0's list lengths already give
    # the per-state visit count.
    visit_counts = np.array([len(p_accum[0][loc]) for loc in range(n_states)], dtype=np.int64)

    # ── Save raw arrays for post-hoc predictive-coding analysis ───────────────
    # p_rates: list of n_f arrays, each (n_states, n_cells_f)
    # Concatenate across frequencies -> (n_states, total_p_cells)
    p_all = np.concatenate(p_rates, axis=1)
    np.save(os.path.join(ep_dir, "p_rates.npy"), p_all)

    g_all = np.concatenate(g_rates, axis=1)
    np.save(os.path.join(ep_dir, "g_rates.npy"), g_all)

    np.save(os.path.join(ep_dir, "visit_counts.npy"), visit_counts)

    # Landmark id -> state id mapping. Landmarks (object ids [0, n_landmarks))
    # are enabled for BOTH conditions (see discritized_objects.py), each
    # placed at exactly one state per environment, so this is meaningful
    # regardless of agent.use_reward - it's what lets post-hoc analysis plot
    # "activity/visits at each value-carrying position" for baseline too, as
    # a control against the reward_modulated condition.
    object_layout = env.environments[0].objects  # (n_states, n_objects) one-hot
    object_ids_per_state = np.argmax(object_layout, axis=1)
    landmark_states = np.full(agent.n_landmarks, -1, dtype=np.int64)
    for sid in range(n_states):
        obj_id = int(object_ids_per_state[sid])
        if obj_id < agent.n_landmarks:
            landmark_states[obj_id] = sid
    np.save(os.path.join(ep_dir, "landmark_states.npy"), landmark_states)

    if agent.use_reward and agent.td is not None:
        # V is keyed by landmark identity (agent.td.V[0] has shape
        # (n_landmarks,)), not by state. Each landmark occupies exactly one
        # state in this environment's layout, so project the table onto
        # those states for spatial plotting; every other (non-landmark)
        # state has no fixed value of its own (its "context" while passing
        # through is whatever landmark was last held, which is path-
        # dependent) and is left as NaN rather than implying a value of 0.
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


def compute_multienv_rates(agent, env, window_steps: int = EVAL_STEPS):
    """Compute place-cell rate maps for every environment in the batch (not
    just env 0) from the trailing `window_steps` of agent.obs_history.

    Pure computation, no file I/O - callers decide where (if anywhere) to
    persist the result. Needed by tem_predictive_analysis.py's reward-zone-
    enrichment analysis, which pools field counts across many environments
    since one environment's state space alone yields too few place fields
    for a meaningful shuffle test.

    Returns
    -------
    rates : dict[str, np.ndarray] or None
        {"env0": (n_states_0, total_p_cells), "env1": ..., ...}, or None if
        there isn't enough history yet.
    env_meta : dict or None
        {"room_widths": [...], "room_depths": [...], "state_densities": [...],
        "reward_location": [x, y]}, aligned with the env indices in `rates`.
    """
    n_hist = len(agent.obs_history)
    if n_hist == 0:
        return None, None

    # Same real-vs-placeholder filtering as run_eval. batch_act advances the
    # whole batch atomically (all_allowed gates every env together - see
    # Whittington2020.batch_act), so env 0 being real implies every other env
    # is real at the same index too.
    real_indices = [i for i, step in enumerate(agent.obs_history) if step[0][0] != -1]
    real_history = [agent.obs_history[i] for i in real_indices]
    real_actions = agent.walk_actions[-len(real_history):]

    n_steps = min(window_steps, len(real_history))
    if n_steps == 0:
        return None, None
    history_slice = real_history[-n_steps:]
    held_indices = real_indices[-n_steps:]
    walk_slice = real_actions[-n_steps:]

    n_envs = agent.batch_size

    # ── Build full-batch model input (every env, not just env 0) ──────────────
    locations_seq = [
        [{"id": step[j][0], "shiny": None} for j in range(n_envs)]
        for step in history_slice
    ]
    obs_seq = np.array(
        [[step[j][1] for j in range(n_envs)] for step in history_slice],
        dtype=np.float32,
    )  # (n_steps, n_envs, n_x)

    v_seq = None
    if agent.use_reward and agent.td is not None:
        v_seq = np.zeros((n_steps, n_envs), dtype=np.float32)
        for j in range(n_envs):
            v_table = agent.td.V[j]
            v_max = float(np.max(v_table))
            for i, idx in enumerate(held_indices):
                key = agent.held_landmark_history[idx][j]
                if key is None:
                    continue
                v_t = float(v_table[key]) if 0 <= key < v_table.shape[0] else 0.0
                v_seq[i, j] = v_t / v_max if v_max > 0 else 0.0

    action_values = agent.step_to_actions(walk_slice)
    action_array = np.reshape(action_values, (n_steps, n_envs))

    model_input = [
        [
            locations_seq[i],
            torch.tensor(obs_seq[i], dtype=torch.float32).to(agent.device),
            [int(a) for a in action_array[i]],
        ]
        for i in range(n_steps)
    ]
    if v_seq is not None:
        for i in range(n_steps):
            model_input[i].append(None)  # td_scale slot (Hebbian gating) - unused
            model_input[i].append(
                torch.tensor(v_seq[i], dtype=torch.float32).to(agent.device)
            )

    # ── Forward pass (no gradient) across the full batch ───────────────────────
    saved_batch_size = agent.tem.hyper.get("batch_size", n_envs)
    agent.tem.eval()
    with torch.no_grad():
        forward = agent.tem(model_input, prev_iter=None)
    agent.tem.train()
    agent.tem.hyper["batch_size"] = saved_batch_size

    # ── Compute rate maps per env ───────────────────────────────────────────────
    n_f = agent.pars["n_f"]
    n_p_list = agent.pars["n_p"]
    n_states_list = agent.n_states

    p_accum = [
        [[[] for _ in range(n_states_list[j])] for _ in range(n_f)]
        for j in range(n_envs)
    ]
    for step in forward:
        for j in range(n_envs):
            loc = step.g[j]["id"]
            if loc < 0:
                continue
            for f in range(n_f):
                p_accum[j][f][loc].append(step.p_inf[f][j].detach().cpu().numpy())

    rates_to_save = {}
    for j in range(n_envs):
        freq_rates_all = []
        for f in range(n_f):
            freq_rates = []
            for loc in range(n_states_list[j]):
                visits = p_accum[j][f][loc]
                if not visits:
                    freq_rates.append(np.zeros(n_p_list[f]))
                else:
                    half = len(visits) // 2
                    usable = visits[half:] if half < len(visits) else visits
                    freq_rates.append(np.mean(usable, axis=0))
            freq_rates_all.append(np.stack(freq_rates, axis=0))
        rates_to_save[f"env{j}"] = np.concatenate(freq_rates_all, axis=1)

    env_meta = {
        "room_widths": list(agent.room_widths),
        "room_depths": list(agent.room_depths),
        "state_densities": list(agent.state_densities),
        "reward_location": agent.reward_location,
    }
    return rates_to_save, env_meta


def _save_probe_snapshot(save_dir, condition, sum_accum, count_accum, n_envs, n_f, agent, episode):
    """Write the current (partial) accumulator state as this condition's
    probe_<condition>_rates.npz/_meta.pkl - overwrites the same filename each
    time, so disk shows one up-to-date snapshot (visible progress, resumable
    if interrupted) rather than accumulating a file per checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    rates_to_save = {}
    for j in range(n_envs):
        freq_rates_all = []
        for f in range(n_f):
            counts = count_accum[j][f]
            sums = sum_accum[j][f]
            means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
            freq_rates_all.append(means.astype(np.float32))
        rates_to_save[f"env{j}"] = np.concatenate(freq_rates_all, axis=1)
    env_meta = {
        "room_widths": list(agent.room_widths),
        "room_depths": list(agent.room_depths),
        "state_densities": list(agent.state_densities),
        "reward_location": agent.reward_location,
        "episode": episode,
    }
    np.savez(os.path.join(save_dir, f"probe_{condition}_rates.npz"), **rates_to_save)
    with open(os.path.join(save_dir, f"probe_{condition}_meta.pkl"), "wb") as fh:
        pickle.dump(env_meta, fh)
    return rates_to_save, env_meta


def run_multienv_probe(agent, env, obs, n_episodes: int, chunk_episodes: int = 10,
                        warmup_steps: int = 500, progress_every: int = 20,
                        progress_prefix: str = "", save_dir: str = None,
                        save_every_episodes: int = None, condition: str = "probe"):
    """Drive a frozen-weight random-policy walk AND accumulate per-env,
    per-frequency place-cell rate maps in one interleaved pass.

    Unlike a "walk fully, then reprocess the whole history" design, this
    function walks one chunk (chunk_episodes rollouts) at a time, forward-
    passes just that chunk through the model (carrying recurrent state
    across chunks via Model.forward's `prev_iter`), accumulates a running
    sum/count per state per frequency module, and then DISCARDS that
    chunk's raw history (agent.obs_history/walk_actions/held_landmark_history
    are reset to empty) before walking the next chunk. Memory stays flat at
    O(n_states) + O(chunk_episodes) regardless of n_episodes, instead of
    growing for the whole walk before being consumed.

    Skips the first `warmup_steps` of accumulation (recurrent path-
    integration state needs a few steps to settle after starting from
    prev_iter=None).

    If save_dir is given, writes a snapshot every save_every_episodes
    episodes via _save_probe_snapshot (overwriting the same file each time)
    so progress is visible on disk as the probe runs, not just held in the
    in-memory accumulator - and the run is resumable from the last snapshot
    if interrupted. The true final result (after n_episodes) is always
    returned and is what callers should treat as authoritative.

    Caller must have already called env.reset() and seeded RNGs; pass the
    resulting `obs` in.
    """
    n_rollout = agent.pars["n_rollout"]
    n_envs = agent.batch_size
    n_f = agent.pars["n_f"]
    n_p_list = agent.pars["n_p"]
    n_states_list = agent.n_states

    sum_accum = [
        [np.zeros((n_states_list[j], n_p_list[f]), dtype=np.float64) for f in range(n_f)]
        for j in range(n_envs)
    ]
    count_accum = [
        [np.zeros(n_states_list[j], dtype=np.int64) for f in range(n_f)]
        for j in range(n_envs)
    ]

    saved_batch_size = agent.tem.hyper.get("batch_size", n_envs)
    agent.tem.eval()
    prev_iter = None
    episode = 0
    steps_processed = 0
    n_total_steps = n_episodes * n_rollout
    n_chunks = (n_episodes + chunk_episodes - 1) // chunk_episodes
    chunk_idx = 0
    start_time = time.monotonic()

    agent.obs_history = []
    agent.walk_actions = []
    agent.held_landmark_history = []

    with torch.no_grad():
        while episode < n_episodes:
            chunk_target = min(episode + chunk_episodes, n_episodes)
            while episode < chunk_target:
                actions = agent.batch_act(obs)
                obs, state, reward = env.step(actions, normalize_step=True)
                if agent.n_walk >= n_rollout:
                    agent.n_walk = 0
                    episode += 1
            chunk_idx += 1

            real_indices = [i for i, step in enumerate(agent.obs_history) if step[0][0] != -1]
            if not real_indices:
                agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []
                continue
            chunk_history = [agent.obs_history[i] for i in real_indices]
            chunk_actions_raw = [agent.walk_actions[i] for i in real_indices]
            chunk_held = [agent.held_landmark_history[i] for i in real_indices]
            chunk_len = len(chunk_history)

            locations_seq = [
                [{"id": step[j][0], "shiny": None} for j in range(n_envs)]
                for step in chunk_history
            ]
            obs_seq = np.array(
                [[step[j][1] for j in range(n_envs)] for step in chunk_history],
                dtype=np.float32,
            )

            v_seq = None
            if agent.use_reward and agent.td is not None:
                v_seq = np.zeros((chunk_len, n_envs), dtype=np.float32)
                for j in range(n_envs):
                    v_table = agent.td.V[j]
                    v_max = float(np.max(v_table))
                    for i in range(chunk_len):
                        key = chunk_held[i][j]
                        if key is None:
                            continue
                        v_t = float(v_table[key]) if 0 <= key < v_table.shape[0] else 0.0
                        v_seq[i, j] = v_t / v_max if v_max > 0 else 0.0

            action_values = agent.step_to_actions(chunk_actions_raw)
            action_array = np.reshape(action_values, (chunk_len, n_envs))

            model_input = [
                [
                    locations_seq[i],
                    torch.tensor(obs_seq[i], dtype=torch.float32).to(agent.device),
                    [int(a) for a in action_array[i]],
                ]
                for i in range(chunk_len)
            ]
            if v_seq is not None:
                for i in range(chunk_len):
                    model_input[i].append(None)  # td_scale slot (Hebbian gating) - unused
                    model_input[i].append(
                        torch.tensor(v_seq[i], dtype=torch.float32).to(agent.device)
                    )

            forward = agent.tem(model_input, prev_iter=prev_iter)

            for local_i, step in enumerate(forward):
                if steps_processed + local_i >= warmup_steps:
                    for j in range(n_envs):
                        loc = step.g[j]["id"]
                        if loc < 0:
                            continue
                        for f in range(n_f):
                            sum_accum[j][f][loc] += step.p_inf[f][j].detach().cpu().numpy()
                            count_accum[j][f][loc] += 1

            prev_iter = [forward[-1]]  # only the last state is needed to continue
            steps_processed += chunk_len

            # Discard this chunk's raw history - already consumed above, and
            # the recurrent model state needed for continuity lives in
            # prev_iter, not in these lists.
            agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

            if save_dir is not None and save_every_episodes and episode % save_every_episodes == 0:
                _save_probe_snapshot(save_dir, condition, sum_accum, count_accum, n_envs, n_f, agent, episode)

            if chunk_idx % progress_every == 0 or episode >= n_episodes:
                elapsed = time.monotonic() - start_time
                rate = steps_processed / elapsed if elapsed > 0 else 0.0
                remaining = (n_total_steps - steps_processed) / rate if rate > 0 else float("nan")
                print(f"  {progress_prefix}ep {episode}/{n_episodes} "
                      f"({steps_processed}/{n_total_steps} steps, {elapsed:.0f}s elapsed, "
                      f"~{remaining:.0f}s remaining)", flush=True)

    agent.tem.train()
    agent.tem.hyper["batch_size"] = saved_batch_size

    rates_to_save, env_meta = _save_probe_snapshot(
        save_dir, condition, sum_accum, count_accum, n_envs, n_f, agent, episode
    ) if save_dir is not None else (None, None)

    if rates_to_save is None:
        rates_to_save = {}
        for j in range(n_envs):
            freq_rates_all = []
            for f in range(n_f):
                counts = count_accum[j][f]
                sums = sum_accum[j][f]
                means = np.divide(
                    sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0
                )
                freq_rates_all.append(means.astype(np.float32))
            rates_to_save[f"env{j}"] = np.concatenate(freq_rates_all, axis=1)
        env_meta = {
            "room_widths": list(agent.room_widths),
            "room_depths": list(agent.room_depths),
            "state_densities": list(agent.state_densities),
            "reward_location": agent.reward_location,
        }

    return rates_to_save, env_meta


def run_eval_multienv(agent, env, episode: int, eval_save_path: str):
    """Thin save-to-disk wrapper around compute_multienv_rates(), writing
    p_rates_multienv.npz + env_meta.pkl into the plots/episode_<N> folder
    run_eval() uses. Kept for callers that want that convention; the
    long-running frozen-weight probe (tem_probe_eval_multienv.py) does NOT
    use this - it saves its (much larger, non-checkpointed) result directly
    under results_sim/predictive_analysis/ instead, via compute_multienv_rates().
    """
    rates, env_meta = compute_multienv_rates(agent, env)
    if rates is None:
        return
    ep_dir = os.path.join(eval_save_path, "plots", f"episode_{episode}")
    os.makedirs(ep_dir, exist_ok=True)
    np.savez(os.path.join(ep_dir, "p_rates_multienv.npz"), **rates)
    with open(os.path.join(ep_dir, "env_meta.pkl"), "wb") as fh:
        pickle.dump(env_meta, fh)
    print(f"  [eval-multienv ep {episode}] saved -> {ep_dir}", flush=True)


def _save_rate_maps(rates, n_cells_list, room_w, room_d, ep_dir, prefix, episode):
    """Save one figure per frequency module.

    Each cell is normalised to its OWN peak (0 = that cell's minimum, 1 =
    that cell's own maximum firing rate) - this matches the convention
    tem_predictive_analysis.py's field detection already uses
    (thresh_frac * that cell's own peak, see is_place_like/detect_fields),
    and is standard for rate-map grids: a single shared scale would wash out
    a weakly-active cell's field shape entirely once a strongly-active cell
    is in the same figure.

    Because normalisation is per-cell, one shared colorbar can only convey
    the *relative* meaning of the color gradient (which is genuinely the
    same for every panel: darkest = that cell's own floor, brightest =
    that cell's own peak) - it can't show the actual numbers, which differ
    per cell. So each panel also gets its own peak-value annotation; the
    colorbar and the annotations together fully specify what a color means.
    """
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
        fig.suptitle(
            f"{label} – {freq_name} – episode {episode}\n"
            f"each panel normalised to its own peak (annotated top-left); "
            f"colorbar shows the shared 0→peak color gradient",
            fontsize=10,
        )

        for j in range(max_cells):
            row, col = j // num_cols, j % num_cols
            cell_map = freq_rates[:, j]          # (n_states,)
            cell_2d = np.reshape(cell_map, (room_d, room_w))
            floor, peak = float(cell_map.min()), float(cell_map.max())
            if peak - floor < MIN_DISPLAY_AMPLITUDE:
                # Genuinely flat/near-inactive cell (common early in training,
                # or a "dead" cell) - normalising to its own near-noise-level
                # range would amplify floating-point/leaky-relu noise into a
                # misleadingly "hot"-looking field. Use a fixed, small range
                # instead so it correctly renders as uniformly dark.
                vmin, vmax = 0.0, MIN_DISPLAY_AMPLITUDE
            else:
                vmin, vmax = floor, peak
            # Each panel gets its own vmin/vmax, so a colorbar built from any
            # one panel's imshow() would show that panel's absolute numbers,
            # not the shared relative scale - normalise every panel to plain
            # [0, 1] instead, and attach a separate, panel-independent
            # ScalarMappable for the colorbar below.
            normed = (cell_2d - vmin) / (vmax - vmin)
            axs[row, col].imshow(normed, origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=1)
            axs[row, col].set_title(f"Cell {j + 1}", fontsize=7)
            axs[row, col].text(
                0.03, 0.94, f"peak {peak:.2f}", transform=axs[row, col].transAxes,
                fontsize=6, color="white", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, lw=0),
            )
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            for spine in axs[row, col].spines.values():
                spine.set_visible(False)

        for j in range(max_cells, num_rows * num_cols):
            axs[j // num_cols, j % num_cols].axis("off")

        # Panel-independent mappable: ticks are plain 0..1, correct for
        # every panel regardless of that cell's own absolute vmin/vmax
        # (which is why the per-panel "peak" annotation carries the actual
        # numbers - this colorbar only conveys the color gradient itself).
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs, shrink=0.6, pad=0.012, aspect=32, ticks=[0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_label("relative activity (0 = each cell's own min, 1 = each cell's own max)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        fname = f"{prefix}_{freq_name}.png"
        fig.savefig(os.path.join(ep_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
