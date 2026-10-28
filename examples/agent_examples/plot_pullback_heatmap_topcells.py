"""Same space-time heatmap layout as plot_pullback_heatmap.py (rows = k
steps before the current reward, columns = episode 0-19 continuous across
the shift, color = activity min-max normalized per row), but built from
check_pullback_overtime_topcells.py's data - activity from only the top 10
most-active cells per position (selected from the real settled rate maps,
independently pre/post), not the whole ~440-cell population. This is the
time-resolved counterpart to shift_effect_topcells_heatmap.png (which only
showed the single pre-vs-post delta, not the within-window time course).

Saved into Thesis/quantitative/loop/overrepresentation_experiment/.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop\overrepresentation_experiment"
EARLY_WINDOW = 10
K_ROWS = 10
LANDMARK_K = {1: 4, 3: 3, 5: 2, 7: 1, 9: 0}

GOLD = "#eda100"
BLUE = "#2a78d6"
GREY = "#898781"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def row_label(k):
    if k == 0:
        return "k=0 (reward)"
    if k in LANDMARK_K:
        return f"k={k} (landmark {LANDMARK_K[k]})"
    return f"k={k}"


def row_tick_color(k):
    if k == 0:
        return GOLD
    if k in LANDMARK_K:
        return BLUE
    return GREY


def load():
    return pd.read_csv(os.path.join(OUT_DIR, "pullback_overtime_topcells_episodes.csv"))


def condition_series(ep_df, condition, k):
    sub = ep_df[ep_df["condition"] == condition]
    pre = sub[sub["era"] == "early_pre"].groupby("episode")[f"k{k}_activity"].mean().reindex(range(EARLY_WINDOW))
    post = sub[sub["era"] == "early_post"].groupby("episode")[f"k{k}_activity"].mean().reindex(range(EARLY_WINDOW))
    return np.concatenate([pre.values, post.values])


def build_grid(ep_df, condition):
    grid = np.full((K_ROWS, 2 * EARLY_WINDOW), np.nan)
    for k in range(K_ROWS):
        grid[k] = condition_series(ep_df, condition, k)
    return grid


def normalize_rows(grid):
    row_min = np.nanmin(grid, axis=1, keepdims=True)
    row_max = np.nanmax(grid, axis=1, keepdims=True)
    span = np.where(row_max - row_min > 1e-12, row_max - row_min, 1.0)
    return (grid - row_min) / span


def plot_pair(cond_a, cond_b, title, out_name):
    ep_df = load()
    grids = {cond: normalize_rows(build_grid(ep_df, cond)) for cond in (cond_a, cond_b)}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), facecolor=PAGE)
    im = None
    for ax, cond in zip(axes, (cond_a, cond_b)):
        im = ax.imshow(grids[cond], aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, origin="upper",
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
    cbar.set_label("top-10-cell activity, normalized per row (0=row min, 1=row max)", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)

    fig.suptitle(title, color=INK_PRIMARY, fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.90,
              "Row = ring position, k steps before the current reward (gold=reward, blue=an actual landmark, "
              "grey=control/filler); activity = mean firing of only the top 10 most-active cells at that position "
              "(from the real settled rate maps, selected independently pre/post). Dotted white = the shift.",
              ha="center", va="top", color=INK_MUTED, fontsize=7.5)
    fig.tight_layout(rect=[0.02, 0.0, 0.98, 0.8])
    fig.subplots_adjust(hspace=0.15)

    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    plot_pair("baseline", "rm", "Space-time top-cell activity (heatmap): baseline vs reward_modulated",
               "pullback_heatmap_topcells_non_lc.png")
    plot_pair("baseline_lc", "rm_lc", "Space-time top-cell activity (heatmap): baseline_lc vs reward_modulated_lc",
               "pullback_heatmap_topcells_lc.png")
    plot_pair("baseline_noshuffle", "rm_noshuffle",
               "Space-time top-cell activity (heatmap): baseline_noshuffle vs reward_modulated_noshuffle",
               "pullback_heatmap_topcells_non_lc_noshuffle.png")
    plot_pair("baseline_lc_noshuffle", "rm_lc_noshuffle",
              "Space-time top-cell activity (heatmap): baseline_lc_noshuffle vs reward_modulated_lc_noshuffle",
              "pullback_heatmap_topcells_lc_noshuffle.png")


if __name__ == "__main__":
    main()
