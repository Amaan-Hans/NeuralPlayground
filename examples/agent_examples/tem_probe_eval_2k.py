"""Frozen-weight 2000-episode probe: load a trained agent and produce the
same checkpoint plots run_eval() makes during training (trajectory,
value_map, object_value_map, place_cells/grid_cells per frequency module),
but built from an entire N-episode frozen walk instead of run_eval's
trailing 500-step window.

Weights are never updated: agent.update() (the backprop + optimizer step
used during training) is never called here - only agent.batch_act() +
env.step() to walk, exactly like the existing tem_probe_eval*.py scripts.
The TD(0) value table (reward_modulated only) is rebuilt fresh at zero by
agent_class(**agent_params) and re-learns online during the walk, same
convention as tem_probe_eval.py - it is bookkeeping the agent already does
inside batch_act(), not a network weight, so freezing "weights" does not
freeze it.

The walk is processed in chunks (chunk_episodes rollouts at a time,
forward-passed immediately, then discarded) so memory stays flat regardless
of walk length - a single forward pass over the FULL window would run out
of GPU memory (the existing multi-env probe hits this wall at ~1200 steps
for a 16-env batch - see run_multienv_probe's docstring in _tem_eval.py).
Recurrent state carries across chunks via Model.forward's prev_iter, so the
result is identical to what one giant forward pass would have produced, had
memory allowed it.

Usage
-----
    cd examples/agent_examples
    python tem_probe_eval_2k.py --run-dir ../../experiments/random/seed_42
    python tem_probe_eval_2k.py --run-dir ../../experiments/random/seed_123 --episodes 2000
"""

import argparse
import importlib.util
import io
import os
import pickle
import random
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tem_eval import _save_rate_maps


class _CPUUnpickler(pickle.Unpickler):
    """Like the default Unpickler, but remaps CUDA tensor storage to CPU.

    The saved 'agent' file is a plain pickle (not torch.save's own zip
    container), so torch.load() can't parse it directly, and plain
    pickle.load (what pd.read_pickle uses) has no map_location - it fails
    outright if a tensor was saved from CUDA and the current node has no
    GPU, since tensor unpickling internally calls
    torch.storage._load_from_bytes, which defaults to the original device.
    This intercepts exactly that step and forces CPU.
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def _load_state_dict(agent_path):
    if torch.cuda.is_available():
        return pd.read_pickle(agent_path)
    with open(agent_path, "rb") as f:
        return _CPUUnpickler(f).load()


def _load_model_from_save(save_path):
    """Load TEM Model class from the copy saved alongside the weights."""
    model_file = os.path.join(save_path, "whittington_2020_model.py")
    spec = importlib.util.spec_from_file_location("tem_model_saved", model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model


def probe_condition(condition, run_dir, n_episodes, chunk_episodes, warmup_steps,
                     probe_seed, progress_every=20):
    use_reward = condition == "reward_modulated"
    save_path = os.path.join(run_dir, condition)
    agent_path = os.path.join(save_path, "agent")

    if not os.path.exists(agent_path):
        print(f"[{condition}] No saved agent at {save_path} - skipping.")
        return

    print(f"\n{'=' * 60}\nProbing: {condition}  ({save_path})\n{'=' * 60}", flush=True)

    # ── Load saved metadata and rebuild agent/env from it ─────────────────────
    training_dict = pd.read_pickle(os.path.join(save_path, "params.dict"))
    agent_params = training_dict["agent_params"]
    hyper = pd.read_pickle(os.path.join(save_path, "agent_hyper"))
    # Handles landing on a node without a visible GPU even though the
    # checkpoint was saved from CUDA tensors (some bigbatch nodes don't
    # expose a working GPU) - see _load_state_dict/_CPUUnpickler above.
    state_dict = _load_state_dict(agent_path)

    ModelClass = _load_model_from_save(save_path)
    tem = ModelClass(hyper)
    tem.load_state_dict(state_dict)
    tem.eval()

    env = training_dict["env_class"](**training_dict["env_params"])
    agent = training_dict["agent_class"](**agent_params)
    agent.tem.load_state_dict(state_dict)
    agent.tem.eval()
    print(f"  TEM weights loaded ({len(state_dict)} tensors)")

    # ── Seed and reset ─────────────────────────────────────────────────────────
    random.seed(probe_seed)
    np.random.seed(probe_seed)
    obs, state = env.reset(random_state=False, custom_state=[0, 0])

    n_rollout = agent.pars["n_rollout"]
    n_f = agent.pars["n_f"]
    n_p_list = agent.pars["n_p"]
    n_g_list = agent.pars["n_g"]
    n_states = agent.n_states[0]
    room_w = int(agent.room_widths[0])
    room_d = int(agent.room_depths[0])

    p_sum = [np.zeros((n_states, n_p_list[f]), dtype=np.float64) for f in range(n_f)]
    p_cnt = [np.zeros(n_states, dtype=np.int64) for f in range(n_f)]
    g_sum = [np.zeros((n_states, n_g_list[f]), dtype=np.float64) for f in range(n_f)]
    g_cnt = [np.zeros(n_states, dtype=np.int64) for f in range(n_f)]
    positions = []

    saved_batch_size = agent.tem.hyper.get("batch_size", 16)
    agent.tem.eval()
    prev_iter = None
    episode = 0
    steps_processed = 0
    n_total_steps = n_episodes * n_rollout
    chunk_idx = 0
    start_time = time.monotonic()

    agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

    print(f"  Walking {n_episodes} episodes ({n_total_steps} steps, no weight updates)...",
          flush=True)

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

            locations_seq = [[{"id": step[0][0], "shiny": None}] for step in chunk_history]
            obs_seq = np.array([step[0][1] for step in chunk_history], dtype=np.float32)

            v_seq = None
            if use_reward and agent.td is not None:
                v_table = agent.td.V[0]
                v_max = float(np.max(v_table))
                v_seq = np.zeros(chunk_len, dtype=np.float32)
                for i in range(chunk_len):
                    key = chunk_held[i][0]
                    if key is None:
                        continue
                    v_t = float(v_table[key]) if 0 <= key < v_table.shape[0] else 0.0
                    v_seq[i] = v_t / v_max if v_max > 0 else 0.0

            action_values = agent.step_to_actions(chunk_actions_raw)
            action_array = np.reshape(action_values, (chunk_len, agent.batch_size))[:, 0]

            model_input = [
                [
                    locations_seq[i],
                    torch.tensor(obs_seq[i:i + 1], dtype=torch.float32).to(agent.device),
                    [int(action_array[i])],
                ]
                for i in range(chunk_len)
            ]
            if v_seq is not None:
                for i in range(chunk_len):
                    model_input[i].append(None)  # td_scale slot (Hebbian gating) - unused
                    model_input[i].append(
                        torch.tensor([v_seq[i]], dtype=torch.float32).to(agent.device)
                    )

            forward = agent.tem(model_input, prev_iter=prev_iter)

            for local_i, step in enumerate(forward):
                global_i = steps_processed + local_i
                loc = step.g[0]["id"]
                if global_i >= warmup_steps and loc >= 0:
                    for f in range(n_f):
                        p_sum[f][loc] += step.p_inf[f][0].detach().cpu().numpy()
                        p_cnt[f][loc] += 1
                        g_sum[f][loc] += step.g_inf[f][0].detach().cpu().numpy()
                        g_cnt[f][loc] += 1

            for step in chunk_history:
                positions.append(step[0][2])

            prev_iter = [forward[-1]]
            steps_processed += chunk_len
            agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

            if chunk_idx % progress_every == 0 or episode >= n_episodes:
                elapsed = time.monotonic() - start_time
                rate = steps_processed / elapsed if elapsed > 0 else 0.0
                remaining = (n_total_steps - steps_processed) / rate if rate > 0 else float("nan")
                print(f"    ep {episode:4d}/{n_episodes}  "
                      f"({steps_processed}/{n_total_steps} steps, {elapsed:.0f}s elapsed, "
                      f"~{remaining:.0f}s remaining)", flush=True)

    agent.tem.train()
    agent.tem.hyper["batch_size"] = saved_batch_size

    p_rates = [
        np.divide(p_sum[f], p_cnt[f][:, None], out=np.zeros_like(p_sum[f]), where=p_cnt[f][:, None] > 0).astype(np.float32)
        for f in range(n_f)
    ]
    g_rates = [
        np.divide(g_sum[f], g_cnt[f][:, None], out=np.zeros_like(g_sum[f]), where=g_cnt[f][:, None] > 0).astype(np.float32)
        for f in range(n_f)
    ]
    visit_counts = p_cnt[0]

    label = f"probe_{n_episodes}"
    ep_dir = os.path.join(save_path, "plots", f"episode_{label}")
    os.makedirs(ep_dir, exist_ok=True)

    p_all = np.concatenate(p_rates, axis=1)
    np.save(os.path.join(ep_dir, "p_rates.npy"), p_all)
    g_all = np.concatenate(g_rates, axis=1)
    np.save(os.path.join(ep_dir, "g_rates.npy"), g_all)
    np.save(os.path.join(ep_dir, "visit_counts.npy"), visit_counts)

    # ── Landmark id -> state id mapping (enabled for both conditions) ─────────
    object_layout = env.environments[0].objects
    object_ids_per_state = np.argmax(object_layout, axis=1)
    landmark_states = np.full(agent.n_landmarks, -1, dtype=np.int64)
    for sid in range(n_states):
        obj_id = int(object_ids_per_state[sid])
        if obj_id < agent.n_landmarks:
            landmark_states[obj_id] = sid
    np.save(os.path.join(ep_dir, "landmark_states.npy"), landmark_states)

    v_per_state = None
    if use_reward and agent.td is not None:
        v_per_state = np.full(n_states, np.nan, dtype=np.float32)
        for sid in range(n_states):
            obj_id = int(object_ids_per_state[sid])
            if obj_id < agent.n_landmarks:
                v_per_state[sid] = agent.td.V[0][obj_id]
        np.save(os.path.join(ep_dir, "v_table.npy"), v_per_state)

    # ── 1. Trajectory (full walk) ───────────────────────────────────────────────
    xs = [float(p[0]) for p in positions]
    ys = [float(p[1]) for p in positions]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, color="steelblue", alpha=0.25, linewidth=0.4)
    ax.scatter(xs[0], ys[0], c="green", s=70, zorder=5, label="start")
    ax.scatter(xs[-1], ys[-1], c="red", s=70, zorder=5, label="end")
    if use_reward:
        rx, ry = agent.reward_location
        ax.scatter([rx], [ry], c="gold", s=200, marker="*", zorder=6, label="reward")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Trajectory – env 0 – {label}\n(full probe walk, {len(positions)} steps)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ep_dir, "trajectory.png"), dpi=150)
    plt.close(fig)

    # ── 2. Value map + object/value overlay (reward condition only) ───────────
    if use_reward and agent.td is not None and n_states == room_d * room_w:
        v_grid = np.reshape(v_per_state, (room_d, room_w))
        cmap = plt.get_cmap("hot").copy()
        cmap.set_bad(color="gray")
        rx, ry = agent.reward_location

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(v_grid, origin="lower", cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label="V(landmark)")
        ax.scatter([rx + room_w / 2 - 0.5], [ry + room_d / 2 - 0.5], c="cyan", s=120,
                   marker="*", zorder=6, label="reward")
        ax.legend(fontsize=7)
        ax.set_title(f"Landmark value V — env 0 – {label}\n"
                     f"(gray = non-landmark state, no fixed value of its own)")
        ax.set_xlabel("x bin")
        ax.set_ylabel("y bin")
        fig.tight_layout()
        fig.savefig(os.path.join(ep_dir, "value_map.png"), dpi=150)
        plt.close(fig)

        obj_grid = np.reshape(object_ids_per_state, (room_d, room_w))
        is_landmark_grid = np.reshape(object_ids_per_state < agent.n_landmarks, (room_d, room_w))

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(v_grid, origin="lower", cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label="V(landmark)")
        for row in range(room_d):
            for col in range(room_w):
                ax.text(col, row, str(int(obj_grid[row, col])),
                         ha="center", va="center", fontsize=6, color="cyan")
                if is_landmark_grid[row, col]:
                    ax.add_patch(mpatches.Rectangle(
                        (col - 0.5, row - 0.5), 1, 1,
                        fill=False, edgecolor="lime", linewidth=2,
                    ))
        ax.scatter([rx + room_w / 2 - 0.5], [ry + room_d / 2 - 0.5], c="cyan", s=120,
                   marker="*", zorder=6, label="reward")
        ax.legend(fontsize=7)
        ax.set_title(
            f"Object id (cyan text) + landmark V – env 0 – {label}\n"
            f"lime boxes = the {agent.n_landmarks} landmark states (ids 0..{agent.n_landmarks - 1})"
        )
        ax.set_xlabel("x bin")
        ax.set_ylabel("y bin")
        fig.tight_layout()
        fig.savefig(os.path.join(ep_dir, "object_value_map.png"), dpi=150)
        plt.close(fig)

    # ── 3 & 4. Place cell and grid cell rate maps (full walk) ─────────────────
    _save_rate_maps(p_rates, n_p_list, room_w, room_d, ep_dir, "place_cells", label)
    _save_rate_maps(g_rates, n_g_list, room_w, room_d, ep_dir, "grid_cells", label)

    print(f"  [{condition}] Done: {ep_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                         help="Path to the experiment root containing baseline/ and "
                              "reward_modulated/ subfolders (e.g. ../../experiments/random/seed_42)")
    parser.add_argument("--episodes", type=int, default=2000,
                         help="Number of frozen-walk episodes (default: 2000).")
    parser.add_argument("--chunk-episodes", type=int, default=10,
                         help="Episodes per forward-pass chunk (default: 10).")
    parser.add_argument("--warmup-steps", type=int, default=500,
                         help="Initial steps excluded from rate-map accumulation "
                              "while recurrent state settles (default: 500).")
    parser.add_argument("--probe-seed", type=int, default=999,
                         help="RNG seed for this probe's own walk (default: 999).")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    for condition in ("baseline", "reward_modulated"):
        probe_condition(
            condition, run_dir,
            n_episodes=args.episodes,
            chunk_episodes=args.chunk_episodes,
            warmup_steps=args.warmup_steps,
            probe_seed=args.probe_seed,
        )

    print(f"\nAll done. Results under: {run_dir}/{{baseline,reward_modulated}}/plots/episode_probe_{args.episodes}/")


if __name__ == "__main__":
    main()
