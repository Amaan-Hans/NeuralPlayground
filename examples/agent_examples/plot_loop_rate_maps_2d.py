"""Sample of place-cell rate maps drawn on the actual 10x10 room grid (not
just a 1D line over ring position), side by side with the TD(0) value
signal mapped onto the same 2D grid - a visual cross-check that spatial
tuning and the learned value both line up with the real ring geometry.

Rate maps: from the already-computed rm_seed42_place_cells.csv (env 0), the
8 highest-range cells (picked on the pre-shift half, same cells reused for
post so the two figures are directly comparable), drawn as a 2D heatmap over
the room instead of a 1D trace - only the 20 ring states have data,
everything else is background (the loop never visits off-ring states).

Value: two panels per figure -
  - "value landmarks only": V(landmark) colored ONLY at the 5 actual
    landmark ring positions (+ the reward state, which inherits the closest
    landmark's value) - everything else left as background. This is the
    direct, undecorated answer to "where does the agent actually have a
    learned value" - nothing is painted onto control/filler states.
  - "value, held-landmark rule": the same V propagated across every ring
    position via "value persists from the last landmark encountered until
    the next one", matching WhittingtonTEM.batch_act's real held_landmark
    bookkeeping (see _value_for_history) - this is what the agent itself
    was actually fed as its value side-channel on every single step, not
    just landmark steps.

V comes from check_value_pullback_full.py's value_pullback_full.csv, which
runs the FULL 500 episodes through the real episode-250 shift and snapshots
V separately at the pre/post boundary and at the end - so the post-shift
panel here is genuine post-shift-learned value, not pre-shift V repainted
onto the new positions (that was an earlier, since-fixed version of this
script).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AGENT_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AGENT_EXAMPLES_DIR)
from probe_loop_ring import LANDMARK_DISTANCES, ROOM_SIZE  # noqa: E402
from plot_loop_layout import xy_to_cell  # noqa: E402

LOOP_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop"
DOWNLOADS = r"C:\Users\hansl\Downloads"
COND = "rm"
SEED = 42
N_RING_STATES = 20
N_CELLS_TO_PLOT = 8
N_COLS = 4
ROOM_N = int(ROOM_SIZE)

# Fallback only - value_pullback_full.csv (from check_value_pullback_full.py)
# is the real source once it exists.
_FALLBACK_V = np.array([0.68788349, 0.75906337, 0.83716684, 0.92290205, 1.06275573])

BACKGROUND = "#e8e7e1"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GOLD = "#eda100"


def load_v_per_half():
    full_path = os.path.join(LOOP_DIR, "value_pullback_full.csv")
    if os.path.exists(full_path):
        df = pd.read_csv(full_path).sort_values("landmark_id")
        return {"pre": df["V_pre"].values, "post": df["V_post"].values}
    print("WARNING: value_pullback_full.csv not found - falling back to pre-shift-only V for both halves "
          "(run check_value_pullback_full.py for genuine post-shift value).")
    return {"pre": _FALLBACK_V, "post": _FALLBACK_V}


def ring_positions_and_categories(half):
    """Re-derive the ring's (x,y) per ring_pos and landmark ring indices for
    the given half ("pre" or "post"), matching build_ring_env_agent()'s /
    apply_shift()'s own geometry exactly (read from the real snapshot in the
    CSV, not re-derived analytically)."""
    df = pd.read_csv(os.path.join(LOOP_DIR, f"{COND}_seed{SEED}_place_cells.csv"))
    df0 = df[(df["env_id"] == 0) & (df["half"] == half)].set_index("ring_pos").reindex(range(N_RING_STATES))
    xy = list(zip(df0["x"], df0["y"]))
    landmark_ring_idx = sorted(df0.index[df0["is_landmark"] == 1].tolist())
    reward_ring_idx = int(df0.index[df0["is_reward"] == 1][0])
    control_ring_idx = df0.index[df0["is_control_landmark"] == 1].tolist()
    return df0, xy, landmark_ring_idx, reward_ring_idx, control_ring_idx


def value_landmarks_only(landmark_ring_idx, reward_ring_idx, v_per_landmark, n_ring_states):
    """V painted ONLY at the 5 real landmark positions + the reward state
    (which inherits the closest landmark's value, id 4 - the last one held
    when reward is reached) - everywhere else is NaN/background."""
    val = np.full(n_ring_states, np.nan)
    for i, pos in enumerate(landmark_ring_idx):
        val[pos] = v_per_landmark[i]
    val[reward_ring_idx] = v_per_landmark[-1]
    return val


def propagate_value(landmark_ring_idx, v_per_landmark, n_ring_states):
    """Value 'held' at every ring position, using the same rule the agent
    itself uses: value stays at whatever landmark was most recently
    encountered (looping around - positions before the first landmark still
    carry the last landmark's value, from the previous lap)."""
    val = np.zeros(n_ring_states)
    cur = v_per_landmark[-1]
    for pos in range(n_ring_states):
        if pos in landmark_ring_idx:
            cur = v_per_landmark[landmark_ring_idx.index(pos)]
        val[pos] = cur
    return val


def grid_from_ring(xy, values_by_pos, fill=np.nan):
    grid = np.full((ROOM_N, ROOM_N), fill)
    for pos, v in enumerate(values_by_pos):
        row, col = xy_to_cell(xy[pos])
        grid[row, col] = v
    return grid


def draw_grid(ax, grid, title, cmap, landmark_ring_idx, reward_ring_idx, control_ring_idx, xy, vmin=None, vmax=None):
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_facecolor(BACKGROUND)
    for pos in landmark_ring_idx:
        row, col = xy_to_cell(xy[pos])
        ax.plot(col, row, "s", mfc="none", mec="#2a78d6", mew=1.6, markersize=10)
    row, col = xy_to_cell(xy[reward_ring_idx])
    ax.plot(col, row, "*", color=GOLD, markersize=13, mec="white", mew=0.6)
    for pos in control_ring_idx:
        row, col = xy_to_cell(xy[pos])
        ax.plot(col, row, "s", mfc="none", mec="#eb6834", mew=1.2, markersize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=INK_PRIMARY, fontsize=9.5, fontweight="bold", pad=6)
    return im


def plot_half(half, v_per_half, chosen_cells=None):
    df0, xy, landmark_ring_idx, reward_ring_idx, control_ring_idx = ring_positions_and_categories(half)
    v_per_landmark = v_per_half[half]

    p_cols = [c for c in df0.columns if c.startswith("p_")]
    if chosen_cells is None:
        ranges = (df0[p_cols].max() - df0[p_cols].min()).sort_values(ascending=False)
        chosen_cells = list(ranges.index[:N_CELLS_TO_PLOT])

    val_masked = value_landmarks_only(landmark_ring_idx, reward_ring_idx, v_per_landmark, N_RING_STATES)
    val_propagated = propagate_value(landmark_ring_idx, v_per_landmark, N_RING_STATES)

    n_rows = int(np.ceil(N_CELLS_TO_PLOT / N_COLS))
    n_cols_total = N_COLS + 1  # rate-map columns + 1 value column (2 stacked value panels)
    fig = plt.figure(figsize=(3.0 * n_cols_total, 3.1 * n_rows), facecolor=PAGE)
    gs = fig.add_gridspec(n_rows, n_cols_total)

    rate_cmap = plt.get_cmap("YlOrRd").copy()
    rate_cmap.set_bad(BACKGROUND)
    for i, cell in enumerate(chosen_cells):
        ax = fig.add_subplot(gs[i // N_COLS, i % N_COLS])
        grid = grid_from_ring(xy, df0[cell].values)
        draw_grid(ax, grid, cell.replace("p_", ""), rate_cmap, landmark_ring_idx, reward_ring_idx, control_ring_idx, xy)

    value_cmap = plt.get_cmap("viridis").copy()
    value_cmap.set_bad(BACKGROUND)
    vmin, vmax = float(v_per_landmark.min()), float(v_per_landmark.max())

    ax_masked = fig.add_subplot(gs[0, N_COLS])
    grid_masked = grid_from_ring(xy, val_masked)
    im_masked = draw_grid(ax_masked, grid_masked, "V, landmarks only", value_cmap,
                           landmark_ring_idx, reward_ring_idx, control_ring_idx, xy, vmin=vmin, vmax=vmax)
    cbar1 = fig.colorbar(im_masked, ax=ax_masked, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors=INK_MUTED, labelsize=6.5)

    ax_prop = fig.add_subplot(gs[1 if n_rows > 1 else 0, N_COLS])
    grid_prop = grid_from_ring(xy, val_propagated)
    im_prop = draw_grid(ax_prop, grid_prop, "V, held-landmark rule", value_cmap,
                         landmark_ring_idx, reward_ring_idx, control_ring_idx, xy, vmin=vmin, vmax=vmax)
    cbar2 = fig.colorbar(im_prop, ax=ax_prop, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors=INK_MUTED, labelsize=6.5)

    legend_items = [
        plt.Line2D([0], [0], marker="s", mfc="none", mec="#2a78d6", mew=1.6, linestyle="none",
                   markersize=9, label="value landmark"),
        plt.Line2D([0], [0], marker="*", color=GOLD, mec="white", linestyle="none", markersize=11, label="reward"),
        plt.Line2D([0], [0], marker="s", mfc="none", mec="#eb6834", mew=1.3, linestyle="none",
                   markersize=8, label="control landmark"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.0))

    half_label = "pre-shift" if half == "pre" else "post-shift"
    v_source = "genuinely re-learned after 250 post-shift episodes" if half == "post" else "learned over 250 pre-shift episodes"
    fig.suptitle(f"2D rate maps vs. value map on the ring ({COND}, seed {SEED}, {half_label})",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.955,
              f"{N_CELLS_TO_PLOT} highest-range place cells (left grid), plotted on the real 10x10 room grid (grey "
              f"= never visited). Right column: TD(0) value ({v_source}) - top shows V painted only at the actual "
              "value-carrying states (5 landmarks + reward), bottom shows the same V propagated using the agent's "
              "real held-landmark persistence rule (value carries over until the next landmark is seen).",
              ha="center", va="top", color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=[0, 0.055, 1, 0.9])

    out_path = os.path.join(DOWNLOADS, f"loop_rate_maps_2d_{half}.png")
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"Saved {out_path}")
    return chosen_cells


def main():
    v_per_half = load_v_per_half()
    chosen_cells = plot_half("pre", v_per_half)
    plot_half("post", v_per_half, chosen_cells=chosen_cells)


if __name__ == "__main__":
    main()
