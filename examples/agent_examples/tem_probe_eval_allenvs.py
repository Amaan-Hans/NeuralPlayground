"""All-environment frozen-weight probe: place-cell rate maps for ALL 16
environments (not just env 0), averaged over the full N-episode walk (all
steps, minus a warmup burn-in) rather than a trailing window. Extends the
existing single-env tem_probe_eval_2k.py the same way run_multienv_probe()
in _tem_eval.py already extends run_eval() to multiple environments - this
script reuses that exact function rather than reimplementing the walk.

Weights are never updated: agent.update() (backprop + optimizer step) is
never called here - only agent.batch_act() + env.step() to walk. TD(0)
value table (reward_modulated only) is rebuilt fresh at zero by
agent_class(**agent_params) and re-learns online during the walk, same
convention as tem_probe_eval.py and tem_probe_eval_2k.py.

Output: one CSV containing every environment x every condition x every
state, in the same column layout as the existing env-0 datasets (seed,
condition, env_id, state_id, x, y, is_landmark, is_reward, value,
p_<freq>_<idx>...) - just with an added env_id column and many more rows
(one row per state per environment per condition, instead of just env 0's
states). Place cells only (matches the existing env-0 datasets - no grid
cells), all 440 place cells (no 150-subset variant here).

Usage
-----
    cd examples/agent_examples
    python tem_probe_eval_allenvs.py --run-dir ../../experiments/random/seed_42 \\
        --seed-label 42 --out ../../experiments/random/datasets/tem_dataset_seed42_allenvs.csv
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
    """Like the default Unpickler, but remaps CUDA tensor storage to CPU -
    see tem_probe_eval_2k.py for the full explanation (some bigbatch nodes
    don't expose a working GPU even though the checkpoint was saved from
    CUDA tensors).
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def _load_pickle_safe(path):
    """Load any of the saved pickle files (agent state_dict, agent_hyper,
    params.dict) safely regardless of whether this node has a GPU - several
    of these files carry CUDA tensors buried inside them (e.g. agent_hyper's
    p_update_mask), not just the state_dict itself.
    """
    if torch.cuda.is_available():
        return pd.read_pickle(path)
    with open(path, "rb") as f:
        return _CPUUnpickler(f).load()


def _load_model_from_save(save_path):
    """Load TEM Model class from the copy saved alongside the weights."""
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

    print(f"\n{'=' * 60}\nProbing (all envs): {condition}  ({save_path})\n{'=' * 60}", flush=True)

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
            "is_landmark": is_landmark,
            "is_reward": (np.arange(n_states_j) == reward_state_id).astype(int),
            "value": v_per_state,
        })
        feat_df = pd.DataFrame(p_all, columns=col_names(n_p_list))
        rows.append(pd.concat([meta_df, feat_df], axis=1))

    print(f"  [{condition}] Done.", flush=True)
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                         help="Path to the experiment root containing baseline/ and "
                              "reward_modulated/ subfolders (e.g. ../../experiments/random/seed_42)")
    parser.add_argument("--seed-label", type=int, required=True,
                         help="Seed value to record in the output CSV's 'seed' column.")
    parser.add_argument("--episodes", type=int, default=2000,
                         help="Number of frozen-walk episodes (default: 2000).")
    parser.add_argument("--chunk-episodes", type=int, default=10,
                         help="Episodes per forward-pass chunk (default: 10).")
    parser.add_argument("--warmup-steps", type=int, default=500,
                         help="Initial steps excluded from rate-map accumulation "
                              "while recurrent state settles (default: 500).")
    parser.add_argument("--probe-seed", type=int, default=999,
                         help="RNG seed for this probe's own walk (default: 999).")
    parser.add_argument("--out", required=True, help="Output CSV path.")
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
