"""Same data as plot_pullback_rows.py, as a space-time heatmap instead of
stacked line plots: rows = ring position (k steps before the current
reward, k=0 reward itself, k=1/3/5/7/9 the landmarks closest-to-furthest,
the rest control/filler), columns = episode (0-9 pre-shift, 10-19
post-shift), color = place-cell activity, min-max normalized PER ROW along
the time axis (0 = that row's own minimum, 1 = that row's own maximum) -
this is what actually makes rows comparable: a raw shared scale lets one
high-magnitude row (e.g. k=8) wash out every other row's relative change,
while the per-row normalization puts every row's own rise/fall on the same
0-1 footing regardless of its absolute activity level. A real pullback
would show as a bright cell appearing later (further right) in deeper rows
than in shallow ones - a diagonal band - rather than every row lighting
up/dimming in the same column.

One figure per {rotation, LC} grouping (4 total), 2 panels each (baseline,
reward_modulated), sharing one colorbar anchored to the right panel. Saved
into Thesis/quantitative/loop/overrepresentation_experiment/.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_pullback_rows import OUT_DIR, EARLY_WINDOW, K_ROWS, LANDMARK_K, row_label, load, condition_series

PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GOLD = "#eda100"
BLUE = "#2a78d6"
GREY = "#898781"


def row_tick_color(k):
    if k == 0:
        return GOLD
    if k in LANDMARK_K:
        return BLUE
    return GREY


def build_grid(ep_df, condition):
    grid = np.full((K_ROWS, 2 * EARLY_WINDOW), np.nan)
    for k in range(K_ROWS):
        _, y = condition_series(ep_df, condition, k)
        grid[k] = y
    return grid


def normalize_rows(grid):
    """Min-max normalize each row independently to [0, 1] along the time
    (x) axis - makes each row's OWN temporal shape comparable regardless of
    its absolute magnitude (k=8's raw spike no longer drowns out every
    other row's relative change), which is what actually matters for
    spotting a lag/pullback pattern across rows."""
    row_min = np.nanmin(grid, axis=1, keepdims=True)
    row_max = np.nanmax(grid, axis=1, keepdims=True)
    span = np.where(row_max - row_min > 1e-12, row_max - row_min, 1.0)
    return (grid - row_min) / span


def plot_pair(cond_a, cond_b, title, out_name):
    ep_df, base_df = load()
    grids = {cond: normalize_rows(build_grid(ep_df, cond)) for cond in (cond_a, cond_b)}
    vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), facecolor=PAGE)
    im = None
    for ax, cond in zip(axes, (cond_a, cond_b)):
        im = ax.imshow(grids[cond], aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, origin="upper",
                        interpolation="nearest")
        ax.axvline(EARLY_WINDOW - 0.5, color="white", linestyle="--", linewidth=1.6, alpha=0.9)
        ax.set_xticks(range(0, 2 * EARLY_WINDOW, 2))
        ax.set_xticklabels(range(0, 2 * EARLY_WINDOW, 2), color=INK_MUTED, fontsize=8)
        ax.set_xlabel("episode (0-9 pre-shift, 10-19 post-shift)", color=INK_SECONDARY, fontsize=9)
        ax.set_title(cond, color=INK_PRIMARY, fontsize=12, fontweight="bold", pad=8)
        ax.set_yticks(range(K_ROWS))
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_yticklabels([row_label(k) for k in range(K_ROWS)], fontsize=8)
    for tick, k in zip(axes[0].get_yticklabels(), range(K_ROWS)):
        tick.set_color(row_tick_color(k))
    axes[1].set_yticklabels([])

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.015)
    cbar.set_label("activity, normalized per row (0=row min, 1=row max)", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)

    fig.suptitle(title, color=INK_PRIMARY, fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, 0.945,
              "Row = ring position, k steps before the current reward (gold label=reward, blue=an actual landmark, "
              "grey=control/filler) - meaning fixed across the shift even though the physical position moves. "
              "Dashed white = the shift. A real pullback = a bright cell appearing later in deeper (grey/blue, "
              "higher-k) rows than shallow ones, i.e. a diagonal band - not every row changing in the same column.",
              ha="center", va="top", color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=[0.02, 0.0, 0.98, 0.9])

    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    plot_pair("baseline", "rm", "Space-time place-cell activity (heatmap): baseline vs reward_modulated",
               "pullback_heatmap_non_lc.png")
    plot_pair("baseline_lc", "rm_lc", "Space-time place-cell activity (heatmap): baseline_lc vs reward_modulated_lc",
               "pullback_heatmap_lc.png")
    plot_pair("baseline_noshuffle", "rm_noshuffle",
               "Space-time place-cell activity (heatmap): baseline_noshuffle vs reward_modulated_noshuffle",
               "pullback_heatmap_non_lc_noshuffle.png")
    plot_pair("baseline_lc_noshuffle", "rm_lc_noshuffle",
              "Space-time place-cell activity (heatmap): baseline_lc_noshuffle vs reward_modulated_lc_noshuffle",
              "pullback_heatmap_lc_noshuffle.png")


if __name__ == "__main__":
    main()
