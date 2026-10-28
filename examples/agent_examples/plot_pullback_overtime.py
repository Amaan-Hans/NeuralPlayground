"""Plot check_pullback_overtime.py's per-episode activity-share log (all 5
seeds): for each condition, 2 panels (10 episodes right after first
placement / 10 episodes right after the shift that immediately follows),
each showing the reward state's share of total ring activity and each of
the 5 landmarks' shares, episode by episode (mean +/- SEM across seeds),
with a dashed horizontal line marking that position's own settled
(whole-half-average, from the real 250-episode main batch) share.

4 figures, same {rotation, LC} grouping as plot_loop_cycle.py: each has
baseline on top, reward_modulated on the bottom, early-pre on the left,
early-post on the right. Saved into
Thesis/quantitative/loop/overrepresentation_experiment/.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop\overrepresentation_experiment"
N_LANDMARKS = 5
EARLY_WINDOW = 10

GOLD = "#eda100"
LANDMARK_CMAP = plt.get_cmap("Blues")
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"

METRICS = ["reward_share"] + [f"landmark{i}_share" for i in range(N_LANDMARKS)]
COLORS = [GOLD] + [LANDMARK_CMAP(0.35 + 0.55 * i / (N_LANDMARKS - 1)) for i in range(N_LANDMARKS)]
LABELS = ["reward"] + [f"landmark {i} ({d} away)" for i, d in enumerate([9, 7, 5, 3, 1])]


def style_axes(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=10.5, fontweight="bold", pad=8)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE_AXIS)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_panel(ax, ep_df, base_df, condition, era, half):
    sub = ep_df[(ep_df["condition"] == condition) & (ep_df["era"] == era)]
    base_sub = base_df[(base_df["condition"] == condition) & (base_df["half"] == half)]

    for metric, color, label in zip(METRICS, COLORS, LABELS):
        stats = sub.groupby("episode")[metric].agg(["mean", "sem"]).reindex(range(EARLY_WINDOW))
        ax.plot(stats.index, stats["mean"], color=color, linewidth=1.8, marker="o", markersize=3, label=label)
        ax.fill_between(stats.index, stats["mean"] - stats["sem"], stats["mean"] + stats["sem"],
                         color=color, alpha=0.15, linewidth=0)
        ax.axhline(base_sub[metric].mean(), color=color, linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlim(0, EARLY_WINDOW - 1)
    ax.set_xlabel("episode (relative to placement)", color=INK_SECONDARY, fontsize=8.5)
    era_label = "first 10 episodes (start)" if era == "early_pre" else "first 10 episodes (after shift)"
    style_axes(ax, f"{condition} - {era_label}")


def plot_pair(cond_a, cond_b, title, out_name):
    ep_df = pd.read_csv(os.path.join(OUT_DIR, "pullback_overtime_episodes.csv"))
    base_df = pd.read_csv(os.path.join(OUT_DIR, "pullback_overtime_baseline.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=PAGE, sharey=True)
    plot_panel(axes[0, 0], ep_df, base_df, cond_a, "early_pre", "pre")
    plot_panel(axes[0, 1], ep_df, base_df, cond_a, "early_post", "post")
    plot_panel(axes[1, 0], ep_df, base_df, cond_b, "early_pre", "pre")
    plot_panel(axes[1, 1], ep_df, base_df, cond_b, "early_post", "post")
    axes[0, 0].set_ylabel("share of total ring activity", color=INK_SECONDARY, fontsize=9)
    axes[1, 0].set_ylabel("share of total ring activity", color=INK_SECONDARY, fontsize=9)
    axes[0, 1].legend(frameon=False, fontsize=7, loc="upper right", labelcolor=INK_SECONDARY)

    fig.suptitle(title, color=INK_PRIMARY, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.955,
              "5 seeds pooled (mean +/- SEM). Solid = per-episode share of total ring-population activity at that "
              "position (reward = gold, landmarks = blue, darker/closer to reward = more saturated). Dashed = that "
              "position's own settled share (from the real 250-episode main batch). 'Overrepresentation then "
              "settling' = reward's solid line starts above its dashed line and decays toward it. 'Pullback' = "
              "landmark lines (esp. closest to reward) rise toward their dashed lines over the 10 episodes.",
              ha="center", va="top", color=INK_MUTED, fontsize=7.5)
    fig.tight_layout(rect=[0, 0.0, 1, 0.88])

    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    plot_pair("baseline", "rm",
               "Place-cell representation over time: baseline vs reward_modulated",
               "pullback_overtime_non_lc.png")
    plot_pair("baseline_lc", "rm_lc",
               "Place-cell representation over time: baseline_lc vs reward_modulated_lc",
               "pullback_overtime_lc.png")
    plot_pair("baseline_noshuffle", "rm_noshuffle",
               "Place-cell representation over time: baseline_noshuffle vs reward_modulated_noshuffle",
               "pullback_overtime_non_lc_noshuffle.png")
    plot_pair("baseline_lc_noshuffle", "rm_lc_noshuffle",
               "Place-cell representation over time: baseline_lc_noshuffle vs reward_modulated_lc_noshuffle",
               "pullback_overtime_lc_noshuffle.png")


if __name__ == "__main__":
    main()
