"""Frozen-weight object-prediction probe with a mid-walk landmark relocation.

Loads the final (episode 5000) trained weights for a given seed/condition,
walks env 0 under a random policy for --episodes episodes with NO weight
updates (agent.update() is never called - only agent.batch_act() +
env.step()), and at --move-episode relocates ONE non-reward value landmark
to a currently-decoy state (a direct swap of the two states' one-hot rows in
env.environments[0].objects - the landmark's object id is unchanged, it just
now sits at a different physical state; agent.held_landmark/TD tracking is
keyed by object id, not state, so it follows the move automatically with no
agent-side changes needed).

"Object prediction" = TEM's own generative pathway's PURE prediction of the
current step's sensory observation, computed from the pre-observation
predicted grid state (transition from g_prev + action, then pattern-
completion retrieval from memory) - i.e. x_logits[2] inside Model.iteration()
(named x_gt_logits in generative()): it never touches the actual current
observation, unlike x_logits[0]/[1] which are reconstructions from the
*inferred* (posterior, already-seen-x) state. This pathway exists identically
whether agent.use_reward is True or False, which is exactly what makes it
comparable across conditions.

Accuracy is tracked two ways:
  - Running (hits, total) counters per (category, window) pair - the summary
    CSV (--out), same as before.
  - The FULL per-step log (--out-raw): one row per step with the exact
    location, predicted object id, actual object id, category, and window -
    small enough (~40,000 rows x 8 narrow columns per condition) that there's
    no need to batch/discard it like the place-cell rate-map probes did; only
    the forward pass itself stays chunked (for GPU memory), unrelated to how
    much we keep on the CPU/output side.
Categories are re-evaluated against whatever object identity currently
occupies each state, so after the move the old landmark location counts as
"decoy" and the new location counts as "value_nonreward" (or whichever
category the moved landmark belongs to).

Also writes:
  --out-layout: the env-0 state_id/x/y/object_id table, captured once before
    the move and once after (a "phase" column distinguishes them).
  --out-moveinfo: one row per seed recording exactly which landmark id moved,
    from which state/coordinates to which.

Usage
-----
    cd examples/agent_examples
    python tem_object_prediction_probe.py --run-dir ../../experiments/random_checkpoints/seed_42 \\
        --seed-label 42 --out ../../experiments/random_checkpoints/datasets/object_pred_seed42.csv \\
        --out-raw ../../experiments/random_checkpoints/datasets/object_pred_seed42_raw.csv \\
        --out-layout ../../experiments/random_checkpoints/datasets/object_pred_seed42_layout.csv \\
        --out-moveinfo ../../experiments/random_checkpoints/datasets/object_pred_seed42_moveinfo.csv
"""

import argparse
import importlib.util
import io
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch

FREQ_NAMES = ["Theta", "Delta", "Beta", "Gamma", "High_Gamma"]


class _CPUUnpickler(pickle.Unpickler):
    """Remaps CUDA tensor storage to CPU when this node has no working GPU -
    see tem_probe_eval_2k.py / tem_probe_eval_allenvs.py for the full
    explanation (several saved pickle files carry CUDA tensors, not just the
    state_dict)."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def _load_pickle_safe(path):
    if torch.cuda.is_available():
        return pd.read_pickle(path)
    with open(path, "rb") as f:
        return _CPUUnpickler(f).load()


def _load_model_from_save(save_path):
    model_file = os.path.join(save_path, "whittington_2020_model.py")
    spec = importlib.util.spec_from_file_location("tem_model_saved", model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model


def _categorize(object_ids_per_state, n_landmarks, n_control, reward_state_id):
    """Return {state_id: category} for env 0, re-derived from whatever object
    identity currently occupies each state (call again after the move)."""
    cats = {}
    for sid, obj_id in enumerate(object_ids_per_state):
        obj_id = int(obj_id)
        if obj_id < n_landmarks:
            cats[sid] = "value_reward" if sid == reward_state_id else "value_nonreward"
        elif obj_id < n_landmarks + n_control:
            cats[sid] = "control"
        else:
            cats[sid] = "decoy"
    return cats


def probe_condition(condition, run_dir, n_episodes, move_episode, chunk_episodes,
                     probe_seed):
    save_path = os.path.join(run_dir, condition)
    agent_path = os.path.join(save_path, "agent")
    if not os.path.exists(agent_path):
        print(f"[{condition}] No saved agent at {save_path} - skipping.")
        return None

    print(f"\n{'=' * 60}\nObject-prediction probe: {condition}  ({save_path})\n{'=' * 60}",
          flush=True)

    training_dict = _load_pickle_safe(os.path.join(save_path, "params.dict"))
    agent_params = training_dict["agent_params"]
    hyper = _load_pickle_safe(os.path.join(save_path, "agent_hyper"))
    state_dict = _load_pickle_safe(agent_path)

    ModelClass = _load_model_from_save(save_path)
    tem = ModelClass(hyper)
    tem.load_state_dict(state_dict)
    tem.eval()

    env = training_dict["env_class"](**training_dict["env_params"])
    agent = training_dict["agent_class"](**agent_params)
    agent.tem.load_state_dict(state_dict)
    agent.tem.eval()
    print(f"  TEM weights loaded ({len(state_dict)} tensors)")

    random.seed(probe_seed)
    np.random.seed(probe_seed)
    obs, state = env.reset(random_state=False, custom_state=[0, 0])

    n_rollout = agent.pars["n_rollout"]
    n_landmarks = agent.n_landmarks
    n_control = getattr(env.environments[0], "n_control_landmarks", 0)

    # ── Reward-nearest state (env 0, 10x10 grid) ──────────────────────────────
    room_w = int(agent.room_widths[0])
    room_d = int(agent.room_depths[0])
    x_array = np.linspace(-room_w / 2 + 0.5, room_w / 2 - 0.5, room_w)
    y_array = np.linspace(-room_d / 2 + 0.5, room_d / 2 - 0.5, room_d)

    def state_xy(sid):
        row, col = sid // room_w, sid % room_w
        return x_array[col], y_array[row]

    all_xy = np.array([state_xy(s) for s in range(room_w * room_d)])
    reward_loc = np.array(agent.reward_location)
    reward_state_id = int(np.argmin(np.linalg.norm(all_xy - reward_loc, axis=1)))

    # ── Full env-0 object layout, before the move ──────────────────────────────
    def layout_snapshot(phase):
        obj_ids = np.argmax(env.environments[0].objects, axis=1)
        cats = _categorize(obj_ids, n_landmarks, n_control, reward_state_id)
        rows = []
        for sid in range(room_w * room_d):
            x, y = state_xy(sid)
            rows.append({
                "phase": phase, "state_id": sid, "x": x, "y": y,
                "object_id": int(obj_ids[sid]), "category": cats[sid],
            })
        return rows

    layout_rows = layout_snapshot("pre")

    # ── Identify the landmark to move (a non-reward value landmark) and its
    # target (a currently-decoy state), then categorize every state ────────────
    object_ids = np.argmax(env.environments[0].objects, axis=1)
    categories = _categorize(object_ids, n_landmarks, n_control, reward_state_id)

    value_nonreward_states = [s for s, c in categories.items() if c == "value_nonreward"]
    decoy_states = [s for s, c in categories.items() if c == "decoy"]
    move_info = None
    if not value_nonreward_states:
        print("  WARNING: no non-reward value landmark found - skipping move.")
        move_from_state = move_to_state = None
    else:
        move_from_state = sorted(value_nonreward_states)[0]
        move_to_state = int(np.random.choice(decoy_states))
        moved_landmark_id = int(object_ids[move_from_state])
        from_x, from_y = state_xy(move_from_state)
        to_x, to_y = state_xy(move_to_state)
        move_info = {
            "moved_object_id": moved_landmark_id,
            "from_state_id": move_from_state, "from_x": from_x, "from_y": from_y,
            "to_state_id": move_to_state, "to_x": to_x, "to_y": to_y,
            "move_episode": move_episode,
        }
        print(f"  Will move landmark id={moved_landmark_id} from state {move_from_state} "
              f"({from_x},{from_y}) to state {move_to_state} ({to_x},{to_y}) "
              f"at episode {move_episode}")

    # ── Counters: (category, window) -> [hits, total] ─────────────────────────
    counters = {}
    raw_records = []

    def record(category, window, correct):
        key = (category, window)
        if key not in counters:
            counters[key] = [0, 0]
        counters[key][0] += int(correct)
        counters[key][1] += 1

    moved = False
    saved_batch_size = agent.tem.hyper.get("batch_size", 16)
    agent.tem.eval()
    prev_iter = None
    episode = 0
    steps_processed = 0
    n_total_steps = n_episodes * n_rollout
    chunk_idx = 0

    agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

    print(f"  Walking {n_episodes} episodes ({n_total_steps} steps, no weight updates)...",
          flush=True)

    with torch.no_grad():
        while episode < n_episodes:
            if not moved and move_from_state is not None and episode >= move_episode:
                objects = env.environments[0].objects
                row_a = objects[move_from_state].copy()
                row_b = objects[move_to_state].copy()
                objects[move_from_state] = row_b
                objects[move_to_state] = row_a
                object_ids = np.argmax(objects, axis=1)
                categories = _categorize(object_ids, n_landmarks, n_control, reward_state_id)
                moved = True
                layout_rows.extend(layout_snapshot("post"))
                print(f"  >> Moved landmark at episode {episode}", flush=True)

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
            if agent.use_reward and agent.td is not None:
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
            action_array = np.reshape(action_values, (chunk_len, 16))[:, 0]

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

            window = "post" if moved else "pre"
            for local_i, step in enumerate(forward):
                loc = step.g[0]["id"]
                if loc < 0:
                    continue
                predicted_id = int(torch.argmax(step.x_logits[2][0]).item())
                actual_id = int(torch.argmax(step.x[0]).item())
                category = categories.get(loc, "decoy")
                record(category, window, predicted_id == actual_id)
                x, y = state_xy(loc)
                raw_records.append({
                    "step": steps_processed + local_i,
                    "episode": (steps_processed + local_i) // n_rollout + 1,
                    "state_id": loc, "x": x, "y": y,
                    "predicted_object_id": predicted_id,
                    "actual_object_id": actual_id,
                    "correct": predicted_id == actual_id,
                    "category": category, "window": window,
                })

            prev_iter = [forward[-1]]
            steps_processed += chunk_len
            agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

            if chunk_idx % 40 == 0 or episode >= n_episodes:
                print(f"    ep {episode:4d}/{n_episodes}  ({steps_processed}/{n_total_steps} steps, "
                      f"window={'post' if moved else 'pre'})", flush=True)

    agent.tem.train()
    agent.tem.hyper["batch_size"] = saved_batch_size

    rows = []
    for (category, window), (hits, total) in counters.items():
        rows.append({
            "condition": condition,
            "category": category,
            "window": window,
            "hits": hits,
            "total": total,
            "accuracy": hits / total if total > 0 else float("nan"),
        })
    summary_df = pd.DataFrame(rows)

    raw_df = pd.DataFrame(raw_records)
    raw_df.insert(0, "condition", condition)

    layout_df = pd.DataFrame(layout_rows)
    layout_df.insert(0, "condition", condition)

    move_df = pd.DataFrame([move_info]) if move_info is not None else pd.DataFrame()
    if not move_df.empty:
        move_df.insert(0, "condition", condition)

    print(f"  [{condition}] Done.", flush=True)
    return summary_df, raw_df, layout_df, move_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed-label", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--move-episode", type=int, default=1500)
    parser.add_argument("--chunk-episodes", type=int, default=10)
    parser.add_argument("--probe-seed", type=int, default=999)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-raw", required=True,
                         help="Per-step predicted-vs-actual log (one row per step).")
    parser.add_argument("--out-layout", required=True,
                         help="Env-0 state/object layout, before and after the move.")
    parser.add_argument("--out-moveinfo", required=True,
                         help="Exactly which object moved, from where to where.")
    args = parser.parse_args()

    if args.move_episode % args.chunk_episodes != 0:
        raise ValueError("--move-episode must be a multiple of --chunk-episodes "
                          "so the move lands cleanly on a chunk boundary.")

    run_dir = os.path.abspath(args.run_dir)
    summary_dfs, raw_dfs, layout_dfs, move_dfs = [], [], [], []
    for condition in ("baseline", "reward_modulated"):
        result = probe_condition(
            condition, run_dir,
            n_episodes=args.episodes,
            move_episode=args.move_episode,
            chunk_episodes=args.chunk_episodes,
            probe_seed=args.probe_seed,
        )
        if result is None:
            continue
        summary_df, raw_df, layout_df, move_df = result
        for df in (summary_df, raw_df, layout_df, move_df):
            if not df.empty:
                df.insert(0, "seed", args.seed_label)
        summary_dfs.append(summary_df)
        raw_dfs.append(raw_df)
        layout_dfs.append(layout_df)
        if not move_df.empty:
            move_dfs.append(move_df)

    for df_list, path in (
        (summary_dfs, args.out),
        (raw_dfs, args.out_raw),
        (layout_dfs, args.out_layout),
        (move_dfs, args.out_moveinfo),
    ):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        pd.concat(df_list, ignore_index=True).to_csv(path, index=False)

    print(f"\nAll done.\n{pd.concat(summary_dfs, ignore_index=True).to_string(index=False)}")
    print(f"-> {args.out}\n-> {args.out_raw}\n-> {args.out_layout}\n-> {args.out_moveinfo}")


if __name__ == "__main__":
    main()
