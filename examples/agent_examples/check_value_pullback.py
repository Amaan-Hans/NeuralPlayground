"""Verify TD(0) value back-up along the ring: does V(landmark) grow larger
the closer that landmark is to the reward, after the agent has looped the
ring many times? This is the "does value pull back toward the reward"
sanity check, run fresh (not from the saved 40-run batch, since
run_probe()/probe_loop_ring_all.py never persisted agent.td.V).

Reuses probe_loop_ring.py's env/agent builder and scripted-walk loop, but
skips the TEM forward pass entirely (place cells / predictions are
irrelevant here) - only agent.batch_act()+env.step() are needed, since
TD(0)'s self.td.update() happens inside batch_act(). Runs only the
PRE-shift 250 episodes (same landmark layout the 40-run batch used for its
pre-shift half) for one seed, for the two reward-using conditions
(reward_modulated, reward_modulated_lc).

Saves a bar chart (V per landmark, ordered 9->7->5->3->1 ring-states from
reward) to the user's Downloads folder.
"""
import os
import random
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

AGENT_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AGENT_EXAMPLES_DIR)
from probe_loop_ring import (  # noqa: E402
    build_ring_env_agent, place_ring_objects, pick_control_states,
    BATCH_SIZE, RING_LOOP, LANDMARK_DISTANCES, TRAJECTORY_SEED, SHIFT_AT_EPISODE,
)

THESIS = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis"
DOWNLOADS = r"C:\Users\hansl\Downloads"
SEED = 42
N_EPISODES = SHIFT_AT_EPISODE  # 250, pre-shift only

CONDITIONS = {
    "rm": os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj", f"seed_{SEED}", "reward_modulated"),
    "rm_lc": os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_lc", f"seed_{SEED}", "reward_modulated_lc"),
}

BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def run_value_buildup(save_path, label):
    cfg = build_ring_env_agent(save_path)
    env, agent = cfg["env"], cfg["agent"]
    n_ring_states = cfg["n_ring_states"]
    n_rollout = agent.pars["n_rollout"]

    assert agent.use_reward and agent.td is not None, f"{label}: agent has no TD value head"

    random.seed(TRAJECTORY_SEED)
    np.random.seed(TRAJECTORY_SEED)
    obs, state = env.reset(random_state=False, custom_state=list(cfg["unique_path_states"][0]))
    control_states = pick_control_states(cfg["non_value_states"], random)
    place_ring_objects(env, cfg, control_states)
    obs = [env.environments[j].make_object_observation(np.array(cfg["unique_path_states"][0])) for j in range(BATCH_SIZE)]
    agent.n_walk = 0
    agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

    loop_idx = 0
    with torch.no_grad():
        for episode in range(N_EPISODES):
            while agent.n_walk < n_rollout:
                agent.batch_act(obs)
                loop_action = RING_LOOP[loop_idx % len(RING_LOOP)]
                env0_state_pre = obs[0][0]
                for i in range(BATCH_SIZE):
                    agent.prev_actions[i] = loop_action
                obs, state, reward = env.step([loop_action] * BATCH_SIZE, normalize_step=True)
                if obs[0][0] != env0_state_pre:
                    loop_idx += 1
            agent.n_walk = 0
            agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []
            if (episode + 1) % 50 == 0:
                print(f"  [{label}] episode {episode + 1}/{N_EPISODES}", flush=True)

    # V is keyed by landmark object id (0..N_RING_LANDMARKS-1), per env slot.
    # Average across the BATCH_SIZE env slots (all identical geometry/seed,
    # so this is just a variance-reduction average, not pooling different
    # conditions).
    v_per_landmark = np.mean(np.stack([agent.td.V[j][:len(LANDMARK_DISTANCES)] for j in range(BATCH_SIZE)]), axis=0)
    print(f"[{label}] V per landmark id (0=furthest..{len(LANDMARK_DISTANCES)-1}=closest): {v_per_landmark}")
    return v_per_landmark


def main():
    results = {label: run_value_buildup(path, label) for label, path in CONDITIONS.items()}

    n_landmarks = len(LANDMARK_DISTANCES)
    x = np.arange(n_landmarks)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=PAGE)
    ax.set_facecolor(SURFACE)
    ax.bar(x - width / 2, results["rm"], width, color=BLUE, label="reward_modulated")
    ax.bar(x + width / 2, results["rm_lc"], width, color=ORANGE, label="reward_modulated_lc")
    ax.set_xticks(x)
    ax.set_xticklabels([f"landmark {i}\n({d} ring-states\nfrom reward)" for i, d in enumerate(LANDMARK_DISTANCES)],
                        fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylabel("learned TD(0) value V(landmark)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("TD value back-up along the ring: does V grow toward the reward?",
                 color=INK_PRIMARY, fontsize=12.5, fontweight="bold", pad=12)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY)
    fig.text(0.5, 0.01,
              f"seed {SEED}, {N_EPISODES} pre-shift episodes, fresh TD(0) table (alpha/gamma from training config). "
              "Monotonic increase landmark 0->4 = value correctly backing up toward reward; flat/noisy = no real backup yet.",
              ha="center", va="bottom", color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = os.path.join(DOWNLOADS, "value_pullback_check.png")
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
