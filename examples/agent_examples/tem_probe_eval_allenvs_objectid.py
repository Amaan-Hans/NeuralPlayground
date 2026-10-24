"""Same frozen-weight all-envs probe as tem_probe_eval_allenvs.py, but also
keeps the raw object id (0-9 = value landmark, 10-19 = control landmark,
30 = decoy, 44 = reward marker, by this project's convention) per state,
instead of collapsing it straight to is_landmark. Needed to test a FAKE
value split defined directly on specific object ids (e.g. "ids 0-4 and
15-19 = fake-valuable, ids 5-14 = not") - something the existing
allenvs_place_cells.csv from training can't do, since it only ever saved
the collapsed is_landmark/is_control_landmark flags, not the id itself.

Usage: same as tem_probe_eval_allenvs.py.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tem_eval import run_multienv_probe

FREQ_NAMES = ["Theta", "Delta", "Beta", "Gamma", "High_Gamma"]


class _CPUUnpickler(pickle.Unpickler):
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


def col_names(n_p_list):
    names = []
    for f, freq in enumerate(FREQ_NAMES):
        for i in range(n_p_list[f]):
            names.append(f"p_{freq}_{i:03d}")
    return names


def probe_condition(condition, run_dir, n_episodes, chunk_episodes, warmup_steps, probe_seed):
    use_reward = condition == "reward_modulated"
    save_path = os.path.join(run_dir, condition)
    agent_path = os.path.join(save_path, "agent")

    if not os.path.exists(agent_path):
        print(f"[{condition}] No saved agent at {save_path} - skipping.")
        return None

    print(f"\n{'=' * 60}\nProbing (all envs, with object id): {condition}  ({save_path})\n{'=' * 60}", flush=True)

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

    rates_to_save, env_meta = run_multienv_probe(
        agent, env, obs,
        n_episodes=n_episodes,
        chunk_episodes=chunk_episodes,
        warmup_steps=warmup_steps,
        progress_every=20,
        progress_prefix=f"[{condition}] ",
        save_dir=None,
        condition=condition,
    )

    n_envs = agent.batch_size
    n_p_list = agent.pars["n_p"]
    room_widths = env_meta["room_widths"]
    room_depths = env_meta["room_depths"]
    reward_location = np.array(env_meta["reward_location"])

    rows = []
    for j in range(n_envs):
        room_w = int(room_widths[j])
        room_d = int(room_depths[j])
        n_states_j = room_w * room_d
        x_array = np.linspace(-room_w / 2 + 0.5, room_w / 2 - 0.5, room_w)
        y_array = np.linspace(-room_d / 2 + 0.5, room_d / 2 - 0.5, room_d)

        def state_xy(sid, x_array=x_array, y_array=y_array, room_w=room_w):
            row, col = sid // room_w, sid % room_w
            return x_array[col], y_array[row]

        all_xy = np.array([state_xy(s) for s in range(n_states_j)])
        reward_state_id = int(np.argmin(np.linalg.norm(all_xy - reward_location, axis=1)))

        object_layout = env.environments[j].objects
        object_ids_per_state = np.argmax(object_layout, axis=1)
        is_landmark = (object_ids_per_state < agent.n_landmarks).astype(int)
        n_control = getattr(env.environments[j], "n_control_landmarks", 0)
        is_control_landmark = (
            (object_ids_per_state >= agent.n_landmarks)
            & (object_ids_per_state < agent.n_landmarks + n_control)
        ).astype(int)

        v_per_state = np.full(n_states_j, np.nan, dtype=np.float32)
        if use_reward and agent.td is not None:
            v_table = agent.td.V[j]
            for sid in range(n_states_j):
                obj_id = int(object_ids_per_state[sid])
                if obj_id < agent.n_landmarks:
                    v_per_state[sid] = v_table[obj_id]

        p_all = rates_to_save[f"env{j}"]  # (n_states_j, total_p_cells)
        xs, ys = zip(*[state_xy(s) for s in range(n_states_j)])

        meta_df = pd.DataFrame({
            "condition": condition,
            "env_id": j,
            "state_id": np.arange(n_states_j),
            "x": xs,
            "y": ys,
            "object_id": object_ids_per_state,
            "is_landmark": is_landmark,
            "is_control_landmark": is_control_landmark,
            "is_reward": (np.arange(n_states_j) == reward_state_id).astype(int),
            "value": v_per_state,
        })
        feat_df = pd.DataFrame(p_all, columns=col_names(n_p_list))
        rows.append(pd.concat([meta_df, feat_df], axis=1))

    print(f"  [{condition}] Done.", flush=True)
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed-label", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--chunk-episodes", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--probe-seed", type=int, default=999)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    dfs = []
    for condition in ("baseline", "reward_modulated"):
        df = probe_condition(
            condition, run_dir,
            n_episodes=args.episodes,
            chunk_episodes=args.chunk_episodes,
            warmup_steps=args.warmup_steps,
            probe_seed=args.probe_seed,
        )
        if df is not None:
            df.insert(0, "seed", args.seed_label)
            dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nAll done. {out.shape} -> {args.out}")


if __name__ == "__main__":
    main()
