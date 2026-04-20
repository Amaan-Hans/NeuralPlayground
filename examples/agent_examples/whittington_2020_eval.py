"""Evaluation script for TEM_rl and TEM_baseline trained models.

Walks each model for up to 1000 steps.  At each checkpoint in CHECKPOINTS,
saves a figure with:
  - Grid cell rate maps (freq 0, viridis)
  - Place cell rate maps (freq 0, hot)
  - Spatial autocorrelation + grid score (GridScorer)
  - V-table reward-propagation heatmap (TEM_rl only)

Figures saved to:
  results_sim/eval_plots/TEM_rl/step_XXXX.png
  results_sim/eval_plots/TEM_baseline/step_XXXX.png
"""

import importlib
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results_sim")
EVAL_DIR     = os.path.join(RESULTS_DIR, "eval_plots")

sys.path.insert(0, PROJECT_ROOT)

from neuralplayground.agents.whittington_2020_extras import (
    whittington_2020_analyse as analyse,
)
from neuralplayground.comparison import GridScorer

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINTS  = [1, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
N_SHOW       = 6   # cells shown per row
N_FREQ       = 5   # TEM frequency modules
FREQ_NAMES   = ["Theta", "Delta", "Beta", "Gamma", "High Gamma"]
EVAL_SEED    = 42  # both agents follow the identical random walk


# ── Model loading ─────────────────────────────────────────────────────────────

def load_tem(model_dir):
    """Load TEM model from saved directory, return (tem, hyper)."""
    spec = importlib.util.spec_from_file_location(
        "saved_model", os.path.join(model_dir, "whittington_2020_model.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    hyper   = pd.read_pickle(os.path.join(model_dir, "agent_hyper"))
    weights = pd.read_pickle(os.path.join(model_dir, "agent"))
    tem = module.Model(hyper)
    tem.load_state_dict(weights)
    tem.eval()
    return tem


# ── Rate-map helpers ──────────────────────────────────────────────────────────

def compute_rate_maps(agent, tem, env):
    """Collect trajectory for env-0, run TEM forward, return (g_rates, p_rates)."""
    model_input, history, environments_raw = agent.collect_final_trajectory()
    environments = [env.collect_environment_info(model_input, history, environments_raw)]
    with torch.no_grad():
        forward = tem(model_input, prev_iter=None)
    g_rates, p_rates = analyse.rate_map(forward, tem, environments)
    return g_rates, p_rates


def get_2d_cell_map(rate_maps, freq, cell_idx, room_w, room_d):
    """Extract a 2-D spatial firing-rate map for one cell.

    rate_maps[env_i][freq] → ndarray (n_locs, n_cells)
    """
    rm = np.asarray(rate_maps[0][freq])   # (n_locs, n_cells)
    n_cells = rm.shape[1]
    if cell_idx >= n_cells:
        return np.zeros((room_w, room_d))
    return rm[:, cell_idx].reshape(room_w, room_d)


# ── Figure saving ─────────────────────────────────────────────────────────────

# 3 sub-rows per frequency module: grid cells, place cells, autocorrelation
_ROWS_PER_FREQ = 3

def save_figure(step, g_rates, p_rates, room_w, room_d, label, is_rl, V_table):
    n_freq_rows = N_FREQ * _ROWS_PER_FREQ          # 15 rows for the 5 modules
    n_rows      = n_freq_rows + (1 if is_rl else 0) # + 1 V-table row for TEM_rl

    fig = plt.figure(figsize=(N_SHOW * 2.2, n_rows * 1.8))
    fig.suptitle(f"{label}  —  step {step:,}", fontsize=12, y=1.005)

    gs = gridspec.GridSpec(n_rows, N_SHOW, figure=fig, hspace=0.6, wspace=0.25)
    scorer = GridScorer(room_w)

    for f, fname in enumerate(FREQ_NAMES):
        base = f * _ROWS_PER_FREQ   # first row index for this frequency block

        # Frequency label spanning all 3 sub-rows on the left margin
        mid_y = 1.0 - (base + _ROWS_PER_FREQ / 2) / n_rows
        fig.text(0.005, mid_y, fname, va="center", ha="left",
                 rotation="vertical", fontsize=8, fontweight="bold")

        for j in range(N_SHOW):
            g_map = get_2d_cell_map(g_rates, f, j, room_w, room_d)
            p_map = get_2d_cell_map(p_rates, f, j, room_w, room_d)

            # Sub-row 0 — grid cells (viridis)
            ax = fig.add_subplot(gs[base, j])
            ax.imshow(g_map, cmap="viridis", interpolation="nearest", origin="lower")
            if j == 0:
                ax.set_ylabel("Grid", fontsize=5, labelpad=2)
            ax.set_title(f"G{j+1}", fontsize=5)
            ax.axis("off")

            # Sub-row 1 — place cells (hot)
            ax = fig.add_subplot(gs[base + 1, j])
            ax.imshow(p_map, cmap="hot", interpolation="nearest", origin="lower")
            if j == 0:
                ax.set_ylabel("Place", fontsize=5, labelpad=2)
            ax.set_title(f"P{j+1}", fontsize=5)
            ax.axis("off")

            # Sub-row 2 — spatial autocorrelation of the grid cell map
            ax = fig.add_subplot(gs[base + 2, j])
            g_map_f = np.nan_to_num(g_map.astype(float), nan=0.0)
            try:
                sac, props = scorer.get_scores(g_map_f)
                gs_val    = props.get("gridscore", np.nan)
                score_str = f"{gs_val:.2f}" if np.isfinite(gs_val) else "n/a"
            except Exception:
                sac       = g_map_f
                score_str = "n/a"
            ax.imshow(sac, cmap="RdBu_r", interpolation="nearest", origin="lower")
            if j == 0:
                ax.set_ylabel("AutoCorr", fontsize=5, labelpad=2)
            ax.set_title(f"GS={score_str}", fontsize=4)
            ax.axis("off")

    # Final row — V-table reward propagation (TEM_rl only)
    if is_rl and V_table is not None:
        ax_v = fig.add_subplot(gs[n_freq_rows, 1 : N_SHOW - 1])
        n_states = room_w * room_d
        # States are y_idx * n_cols + x_idx, so reshape as (n_rows, n_cols) = (room_d, room_w)
        v_grid   = V_table[0, :n_states].reshape(room_d, room_w)
        im = ax_v.imshow(v_grid, cmap="plasma", interpolation="nearest", origin="lower")
        ax_v.plot(room_w // 2, room_d // 2, "w*", markersize=10)
        ax_v.set_title("Discounted value V(s)  [★ = reward location]", fontsize=8)
        ax_v.axis("off")
        fig.colorbar(im, ax=ax_v, fraction=0.046, pad=0.04)
        fig.text(0.005, 0.5 / n_rows, "V-table", va="center", ha="left",
                 rotation="vertical", fontsize=8, fontweight="bold")

    out_dir  = os.path.join(EVAL_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"step_{step:04d}.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved  {os.path.relpath(out_path, PROJECT_ROOT)}")


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(label, model_dir):
    is_rl = label == "TEM_rl"
    print(f"\n{'='*55}")
    print(f"  Evaluating  {label}")
    print(f"{'='*55}")

    tem = load_tem(model_dir)
    training_dict = pd.read_pickle(os.path.join(model_dir, "params.dict"))

    agent = training_dict["agent_class"](**training_dict["agent_params"])
    env   = training_dict["env_class"](**training_dict["env_params"])

    room_w = agent.room_widths[0]
    room_d = agent.room_depths[0]

    # Reset V_table so we watch reward propagation build from scratch
    if is_rl:
        agent._init_td()

    # Fix the numpy seed so both agents follow the exact same random walk.
    # action_policy() uses np.random.choice exclusively, and the environment
    # structure (grid, obstacles) is identical, so rejection patterns match too.
    np.random.seed(EVAL_SEED)
    observation, _state = env.reset(random_state=False, custom_state=[0.5, 0.5])

    checkpoint_set = set(CHECKPOINTS)
    max_steps = max(CHECKPOINTS)
    last_saved = -1

    # Walk and snapshot at each checkpoint
    while agent.n_walk < max_steps:
        action = agent.batch_act(observation)
        observation, _state, _reward = env.step(action, normalize_step=True)

        step = agent.n_walk
        if step not in checkpoint_set or step == last_saved:
            continue
        last_saved = step

        print(f"  Checkpoint {step}/{max_steps}")

        # Update V_table incrementally for TEM_rl (TD on all steps so far)
        V_snapshot = None
        if is_rl:
            history = agent.obs_history[-max(step, 1):]
            agent._update_td_values(history)
            V_snapshot = agent.V_table.copy()

        g_rates, p_rates = compute_rate_maps(agent, tem, env)
        save_figure(step, g_rates, p_rates, room_w, room_d, label, is_rl, V_snapshot)

    print(f"  {label} done.")


if __name__ == "__main__":
    evaluate("TEM_rl",       os.path.join(RESULTS_DIR, "TEM_rl"))
    evaluate("TEM_baseline", os.path.join(RESULTS_DIR, "TEM_baseline"))
    print(f"\nAll evaluations complete. Figures saved to:\n  {EVAL_DIR}")
