"""Space-time view of check_pullback_overtime.py's data: x-axis = time
(episode, continuous across the shift), one row per ring position (k steps
before the CURRENT reward, k=0 reward itself, k=1/3/5/7/9 the landmarks
closest-to-furthest, the rest control/filler, k=19 the reserved-open slot),
y-axis = raw average place-cell activity at that position. All rows share
the same y-axis scale (per figure), so a genuine backward "pullback" would
show up as: the k=1 row's activity rising earliest, k=3 next, ... k=9
latest/least - a visible lag structure going down the rows, not just
everything moving together.

A dashed vertical line marks the shift (between episode 9 and 10 - the
shift happens immediately after the first 10 episodes in this short probe,
not at the usual episode 250). A thin dashed horizontal line per row marks
that position's own settled (whole-half-average, real 250-episode) level -
drawn separately for the pre segment (episodes 0-9) and post segment
(episodes 10-19), since the settled level itself can differ across the
shift.

One figure per {rotation, LC} grouping (4 total, matching
plot_pullback_overtime.py's convention): 2 columns (baseline left, rm
right), K_ROWS rows, all panels sharing one y-axis range for direct
magnitude comparison. Saved into
Thesis/quantitative/loop/overrepresentation_experiment/.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop\overrepresentation_experiment"
EARLY_WINDOW = 10
K_ROWS = 10  # reward (k=0) + the 9 states leading up to it - the whole approach segment,
             # every state not just the 5 landmarks, so the gaps between them are visible too.
             # Set to 20 to cover the full ring (the far side, after reward, too).

GOLD = "#eda100"
BLUE = "#2a78d6"
GREY = "#898781"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"

LANDMARK_K = {1: 4, 3: 3, 5: 2, 7: 1, 9: 0}  # k -> landmark id (distance-based)


def row_label(k):
    if k == 0:
        return "k=0 (reward)"
    if k in LANDMARK_K:
        return f"k={k} (landmark {LANDMARK_K[k]})"
    return f"k={k}"


def row_color(k):
    if k == 0:
        return GOLD
    if k in LANDMARK_K:
        return BLUE
    return GREY


def load():
    ep_df = pd.read_csv(os.path.join(OUT_DIR, "pullback_overtime_episodes.csv"))
    base_df = pd.read_csv(os.path.join(OUT_DIR, "pullback_overtime_baseline.csv"))
    return ep_df, base_df


def condition_series(ep_df, condition, k):
    sub = ep_df[ep_df["condition"] == condition]
    pre = sub[sub["era"] == "early_pre"].groupby("episode")[f"k{k}_activity"].mean().reindex(range(EARLY_WINDOW))
    post = sub[sub["era"] == "early_post"].groupby("episode")[f"k{k}_activity"].mean().reindex(range(EARLY_WINDOW))
    x = np.arange(2 * EARLY_WINDOW)
    y = np.concatenate([pre.values, post.values])
    return x, y


def baseline_levels(base_df, condition, k):
    sub = base_df[base_df["condition"] == condition]
    pre = sub[sub["half"] == "pre"][f"k{k}_activity"].mean()
    post = sub[sub["half"] == "post"][f"k{k}_activity"].mean()
    return pre, post


def plot_pair(cond_a, cond_b, title, out_name):
    ep_df, base_df = load()

    all_vals = []
    for cond in (cond_a, cond_b):
        for k in range(K_ROWS):
            _, y = condition_series(ep_df, cond, k)
            all_vals.append(y)
    all_vals = np.concatenate(all_vals)
    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
    pad = 0.08 * (ymax - ymin)
    ylim = (ymin - pad, ymax + pad)

    fig, axes = plt.subplots(K_ROWS, 2, figsize=(11, 1.15 * K_ROWS), facecolor=PAGE, sharex=True, sharey=True)

    for col, cond in enumerate((cond_a, cond_b)):
        for k in range(K_ROWS):
            ax = axes[k, col]
            x, y = condition_series(ep_df, cond, k)
            color = row_color(k)
            ax.plot(x, y, color=color, linewidth=1.5, marker="o", markersize=2.2)
            pre_base, post_base = baseline_levels(base_df, cond, k)
            ax.hlines(pre_base, 0, EARLY_WINDOW - 1, color=color, linestyle="--", linewidth=0.9, alpha=0.6)
            ax.hlines(post_base, EARLY_WINDOW, 2 * EARLY_WINDOW - 1, color=color, linestyle="--", linewidth=0.9, alpha=0.6)
            ax.axvline(EARLY_WINDOW - 0.5, color=INK_MUTED, linestyle=":", linewidth=1.0, alpha=0.7)
            ax.set_ylim(*ylim)
            ax.set_facecolor(SURFACE)
            ax.tick_params(colors=INK_MUTED, labelsize=6.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(BASELINE_AXIS)
            ax.spines["bottom"].set_color(BASELINE_AXIS)
            ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(row_label(k), color=INK_SECONDARY, fontsize=7, rotation=0, ha="right", va="center")
            if k == 0:
                ax.set_title(cond, color=INK_PRIMARY, fontsize=11, fontweight="bold", pad=10)
            if k == K_ROWS - 1:
                ax.set_xlabel("episode (0-9 pre-shift, 10-19 post-shift)", color=INK_SECONDARY, fontsize=8)

    fig.suptitle(title, color=INK_PRIMARY, fontsize=13, fontweight="bold", y=0.998)
    fig.text(0.5, 0.988,
              "Each row = one ring position, k steps before the current reward (gold=reward, blue=an actual "
              "landmark, grey=control/filler) - meaning stays fixed across the shift even though the physical "
              "position moves. All rows share one y-axis (raw mean place-cell activity). Dashed = that position's "
              "settled (250-episode) level, separately for pre/post. Dotted vertical = the shift. A real pullback "
              "shows as activity rising earlier in shallow-k rows than in deep-k rows, not all rows moving in sync.",
              ha="center", va="top", color=INK_MUTED, fontsize=7.5)
    fig.tight_layout(rect=[0.09, 0.0, 1, 0.975])
    fig.subplots_adjust(hspace=0.15)

    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    plot_pair("baseline", "rm", "Space-time place-cell activity: baseline vs reward_modulated",
               "pullback_rows_non_lc.png")
    plot_pair("baseline_lc", "rm_lc", "Space-time place-cell activity: baseline_lc vs reward_modulated_lc",
               "pullback_rows_lc.png")
    plot_pair("baseline_noshuffle", "rm_noshuffle",
               "Space-time place-cell activity: baseline_noshuffle vs reward_modulated_noshuffle",
               "pullback_rows_non_lc_noshuffle.png")
    plot_pair("baseline_lc_noshuffle", "rm_lc_noshuffle",
               "Space-time place-cell activity: baseline_lc_noshuffle vs reward_modulated_lc_noshuffle",
               "pullback_rows_lc_noshuffle.png")


if __name__ == "__main__":
    main()
