"""Post-hoc analysis: do reward-condition place cells become predictive of reward?

Run this after both training runs have finished.

Produces three comparisons across episode checkpoints:

1. Population activity map
   Mean firing across all place cells per grid state, plotted on the 2D arena.
   Reward condition: activity focus should spread backward from reward over training.

2. Value-cell correlation (reward condition only)
   Pearson correlation between each state's mean place cell activity and V(s).
   Grows over episodes if cells become predictive of future reward.

3. Peak firing distance from reward
   For each cell, find the state with peak firing and compute its Euclidean
   distance to the reward location.  Compares reward vs baseline to show whether
   reward condition cells develop peaks further from the reward site (backward
   shift) while baseline cells stay random.

Usage
-----
    cd examples/agent_examples
    python tem_predictive_analysis.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neuralplayground.comparison import GridScorer

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS_ROOT  = os.path.join(os.getcwd(), "results_sim")
BASELINE_DIR  = os.path.join(RESULTS_ROOT, "baseline",         "plots")
REWARD_DIR    = os.path.join(RESULTS_ROOT, "reward_modulated", "plots")
OUT_DIR       = os.path.join(RESULTS_ROOT, "predictive_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

REWARD_LOCATION = np.array([3.0, 3.0])
ROOM_W, ROOM_D  = 10, 10          # env 0: [-5,5] x [-5,5], state_density=1
STATE_DENSITY   = 1
N_STATES        = ROOM_W * ROOM_D  # 100

# Grid cell centres (matches DiscreteObjectEnvironment xy_combination for env 0)
_x = np.linspace(-ROOM_W/2 + 0.5/STATE_DENSITY, ROOM_W/2 - 0.5/STATE_DENSITY, ROOM_W)
_y = np.linspace(-ROOM_D/2 + 0.5/STATE_DENSITY, ROOM_D/2 - 0.5/STATE_DENSITY, ROOM_D)
XY = np.stack(np.meshgrid(_x, _y), axis=-1)          # (ROOM_D, ROOM_W, 2)
XY_FLAT = XY.reshape(-1, 2)                           # (N_STATES, 2)  row = state id

# Distance from each state to the reward location
DIST_TO_REWARD = np.linalg.norm(XY_FLAT - REWARD_LOCATION, axis=1)  # (N_STATES,)
# ──────────────────────────────────────────────────────────────────────────────


def _episode_dirs(plots_dir):
    """Return sorted list of (episode_int, path) for every episode_N folder."""
    entries = []
    if not os.path.isdir(plots_dir):
        return entries
    for name in os.listdir(plots_dir):
        if name.startswith("episode_"):
            try:
                ep = int(name.split("_")[1])
                entries.append((ep, os.path.join(plots_dir, name)))
            except ValueError:
                pass
    return sorted(entries)


def _load_p_rates(ep_path):
    """Load p_rates.npy -> (N_STATES, n_cells) or None."""
    fpath = os.path.join(ep_path, "p_rates.npy")
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)          # (N_STATES, total_place_cells)


def _load_v_table(ep_path):
    """Load v_table.npy -> (N_STATES,) or None."""
    fpath = os.path.join(ep_path, "v_table.npy")
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)


# ── Analysis 1: Population activity map ───────────────────────────────────────

def plot_population_activity_maps():
    """For each condition, tile population-activity heatmaps across episodes."""
    for label, plots_dir in [("baseline", BASELINE_DIR), ("reward_modulated", REWARD_DIR)]:
        eps = _episode_dirs(plots_dir)
        if not eps:
            print(f"No episode folders found in {plots_dir}")
            continue

        n = len(eps)
        ncols = min(5, n)
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(nrows, ncols,
                                figsize=(ncols * 3, nrows * 3 + 0.5))
        axs = np.array(axs).reshape(nrows, ncols)
        fig.suptitle(f"Population activity map — {label}", fontsize=13)

        vmax_global = 0
        maps = []
        for ep, ep_path in eps:
            p = _load_p_rates(ep_path)
            if p is None:
                maps.append(None)
                continue
            act = p.mean(axis=1)                        # (N_STATES,)
            maps.append((ep, act))
            vmax_global = max(vmax_global, act.max())

        for idx, item in enumerate(maps):
            row, col = divmod(idx, ncols)
            ax = axs[row, col]
            if item is None:
                ax.axis("off")
                continue
            ep, act = item
            grid = act.reshape(ROOM_D, ROOM_W)
            im = ax.imshow(grid, origin="lower", cmap="hot",
                           vmin=0, vmax=vmax_global, aspect="auto")
            # Mark reward location
            rx = (REWARD_LOCATION[0] + ROOM_W/2) / ROOM_W * ROOM_W - 0.5
            ry = (REWARD_LOCATION[1] + ROOM_D/2) / ROOM_D * ROOM_D - 0.5
            ax.plot(rx, ry, "*", color="cyan", markersize=10)
            ax.set_title(f"ep {ep}", fontsize=8)
            ax.axis("off")

        for idx in range(len(maps), nrows * ncols):
            row, col = divmod(idx, ncols)
            axs[row, col].axis("off")

        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f"population_activity_{label}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved: {fname}")


# ── Analysis 2: Value-cell correlation over episodes ──────────────────────────

def plot_value_correlation():
    """Correlation between mean place cell activity per state and V(s).
    Only possible for reward_modulated (V table only saved there).
    """
    eps = _episode_dirs(REWARD_DIR)
    episodes, correlations = [], []

    for ep, ep_path in eps:
        p = _load_p_rates(ep_path)
        v = _load_v_table(ep_path)
        if p is None or v is None:
            continue
        mean_act = p.mean(axis=1)                       # (N_STATES,)
        # Pearson correlation
        if mean_act.std() < 1e-8 or v.std() < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(mean_act, v)[0, 1])
        episodes.append(ep)
        correlations.append(corr)

    if not episodes:
        print("No V table data found — skipping value correlation plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(episodes, correlations, "o-", color="darkorange", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Pearson r  (mean place activity vs V(s))")
    ax.set_title("Place cell–value correlation over training\n(reward_modulated, env 0)")
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "value_correlation.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# ── Analysis 3: Peak firing distance from reward ──────────────────────────────

def plot_peak_distance():
    """Distribution of each cell's peak-firing distance from reward,
    for both conditions across episodes.  A backward shift shows up as
    reward_modulated cells developing peaks at LARGER distances (further back
    along the approach path) compared to baseline.
    """
    results = {}
    for label, plots_dir in [("baseline", BASELINE_DIR), ("reward_modulated", REWARD_DIR)]:
        eps = _episode_dirs(plots_dir)
        ep_list, mean_dist, median_dist = [], [], []
        for ep, ep_path in eps:
            p = _load_p_rates(ep_path)
            if p is None:
                continue
            peak_states = np.argmax(p, axis=0)          # (n_cells,)
            distances   = DIST_TO_REWARD[peak_states]   # (n_cells,)
            ep_list.append(ep)
            mean_dist.append(distances.mean())
            median_dist.append(np.median(distances))
        results[label] = (ep_list, mean_dist, median_dist)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    colours = {"baseline": "steelblue", "reward_modulated": "darkorange"}

    for ax, metric, title in zip(axes, ["mean", "median"],
                                 ["Mean", "Median"]):
        for label, (ep_list, mean_d, med_d) in results.items():
            vals = mean_d if metric == "mean" else med_d
            if ep_list:
                ax.plot(ep_list, vals, "o-", color=colours[label],
                        label=label, linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Distance from reward (grid units)")
        ax.set_title(f"{title} peak-firing distance from reward")
        ax.legend()

    fig.suptitle("Backward shift: place cell peaks vs reward location (env 0)", fontsize=12)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "peak_distance_from_reward.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


    # Also save per-episode histogram comparison at first and last checkpoint
    for label, plots_dir in [("baseline", BASELINE_DIR), ("reward_modulated", REWARD_DIR)]:
        eps = _episode_dirs(plots_dir)
        checkpoints = [eps[0], eps[-1]] if len(eps) >= 2 else eps
        if not checkpoints:
            continue
        fig, axs = plt.subplots(1, len(checkpoints),
                                figsize=(5 * len(checkpoints), 4), sharey=False)
        if len(checkpoints) == 1:
            axs = [axs]
        for ax, (ep, ep_path) in zip(axs, checkpoints):
            p = _load_p_rates(ep_path)
            if p is None:
                ax.axis("off")
                continue
            peak_states = np.argmax(p, axis=0)
            distances   = DIST_TO_REWARD[peak_states]
            ax.hist(distances, bins=20, color=colours[label], edgecolor="white", alpha=0.85)
            ax.axvline(DIST_TO_REWARD[np.argmin(DIST_TO_REWARD)], color="red",
                       linestyle="--", label="reward state")
            ax.set_xlabel("Distance from reward (grid units)")
            ax.set_ylabel("Cell count")
            ax.set_title(f"Episode {ep}")
            ax.legend(fontsize=7)
        fig.suptitle(f"Peak firing distance distribution — {label}", fontsize=11)
        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f"peak_distance_hist_{label}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved: {fname}")


# ── Analysis 4: Mean grid score over training ─────────────────────────────────

def _load_g_rates(ep_path):
    """Load g_rates.npy -> (N_STATES, total_g_cells) or None."""
    fpath = os.path.join(ep_path, "g_rates.npy")
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)


def plot_grid_scores():
    """Mean grid score across all grid cells for both conditions over training.

    For each checkpoint and each condition, every cell's rate map is reshaped
    to (ROOM_W, ROOM_D) and scored with GridScorer.  The mean score across
    all cells is plotted as a time series.  A rising trend in the reward
    condition (or difference from baseline) indicates reward-modulated grid
    cell refinement.
    """
    scorer = GridScorer(ROOM_W)
    colours = {"baseline": "steelblue", "reward_modulated": "darkorange"}

    fig, ax = plt.subplots(figsize=(8, 4))

    for label, plots_dir in [("baseline", BASELINE_DIR), ("reward_modulated", REWARD_DIR)]:
        eps = _episode_dirs(plots_dir)
        ep_list, mean_scores = [], []

        for ep, ep_path in eps:
            g = _load_g_rates(ep_path)
            if g is None:
                continue

            n_cells = g.shape[1]
            scores = []
            for c in range(n_cells):
                cell_map = g[:, c].reshape(ROOM_D, ROOM_W).astype(float)
                if cell_map.max() < 1e-8:
                    continue
                try:
                    _, props = scorer.get_scores(cell_map)
                    scores.append(props["gridscore"])
                except Exception:
                    pass

            if scores:
                ep_list.append(ep)
                mean_scores.append(float(np.mean(scores)))

        if ep_list:
            ax.plot(ep_list, mean_scores, "o-", color=colours[label],
                    label=label, linewidth=2)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean grid score")
    ax.set_title("Mean grid cell score over training (env 0)")
    ax.legend()
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "grid_scores.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# ── Run all analyses ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running predictive-coding analysis...")
    plot_population_activity_maps()
    plot_value_correlation()
    plot_peak_distance()
    plot_grid_scores()
    print(f"\nAll plots saved to: {OUT_DIR}")
