"""Loop cycle plot: object-prediction accuracy as a function of position
around the 20-state ring, pooled across 5 seeds, pre-shift (solid) vs
post-shift (dashed), baseline vs reward_modulated (blue vs orange) overlaid
- one figure per {rotation, LC} pair (4 total), each with an "overall"
panel (every category pooled) plus landmark/control/filler panels. Saved
into Thesis/quantitative/loop/.

Uses probe_loop_ring.py's *_position_accuracy.csv output (per-step
position resolution, correct/total counts per (half, ring_pos, category)).
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOOP_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop"
SEEDS = [42, 7, 1, 2, 3]
N_RING_STATES = 20

BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"

CATEGORIES = ["landmark", "control", "filler"]


def style_axes(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11.5, fontweight="bold", pad=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE_AXIS)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def load_pooled_position_accuracy(cond):
    """Sum correct/total counts across all 5 seeds (not averaging accuracy
    directly), then derive accuracy from the pooled counts - correct for
    unequal per-seed/per-position sample sizes."""
    frames = []
    for seed in SEEDS:
        df = pd.read_csv(os.path.join(LOOP_DIR, f"{cond}_seed{seed}_position_accuracy.csv"))
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    pooled = all_df.groupby(["half", "ring_pos", "category"])[["correct", "total"]].sum().reset_index()
    pooled["accuracy"] = pooled["correct"] / pooled["total"]
    # also compute an "overall" pseudo-category per (half, ring_pos): every category pooled together
    overall = pooled.groupby(["half", "ring_pos"])[["correct", "total"]].sum().reset_index()
    overall["category"] = "overall"
    overall["accuracy"] = overall["correct"] / overall["total"]
    return pd.concat([pooled, overall], ignore_index=True)


def plot_line(ax, df, category, half, color, linestyle, label):
    sub = df[(df["category"] == category) & (df["half"] == half)].set_index("ring_pos").reindex(range(N_RING_STATES))
    # Wilson-ish SEM approximation for a binomial proportion, good enough for a visual band.
    p, n = sub["accuracy"], sub["total"].clip(lower=1)
    sem = np.sqrt(p * (1 - p) / n)
    ax.plot(sub.index, p, color=color, linestyle=linestyle, linewidth=1.8, alpha=0.9,
            marker="o", markersize=3, label=label)
    ax.fill_between(sub.index, p - sem, p + sem, color=color, alpha=0.12, linewidth=0)


def plot_pair(baseline_cond, rm_cond, title, out_name):
    df_bl = load_pooled_position_accuracy(baseline_cond)
    df_rm = load_pooled_position_accuracy(rm_cond)

    panels = [("overall", "Overall (all categories)")] + [(c, c.capitalize()) for c in CATEGORIES]
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.5), facecolor=PAGE, sharey=True)
    for ax, (cat, subtitle) in zip(axes, panels):
        plot_line(ax, df_bl, cat, "pre", BLUE, "-", f"{baseline_cond} (pre-shift)")
        plot_line(ax, df_bl, cat, "post", BLUE, "--", f"{baseline_cond} (post-shift)")
        plot_line(ax, df_rm, cat, "pre", ORANGE, "-", f"{rm_cond} (pre-shift)")
        plot_line(ax, df_rm, cat, "post", ORANGE, "--", f"{rm_cond} (post-shift)")
        ax.set_xlabel("position around the ring (state index)", color=INK_SECONDARY, fontsize=9)
        ax.set_xlim(0, N_RING_STATES - 1)
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, fontsize=6.5, loc="lower center", labelcolor=INK_SECONDARY, ncol=2)
        style_axes(ax, subtitle)
    axes[0].set_ylabel("x_gt accuracy (object identity)", color=INK_SECONDARY, fontsize=9)

    fig.suptitle(title, color=INK_PRIMARY, fontsize=13, fontweight="bold")
    fig.text(
        0.01, 0.01,
        "5 seeds pooled (counts summed, not per-seed-averaged), frozen weights, scripted 20-state ring loop "
        "(border of the inner 6x6 region, uniform 10x10 rooms), per-step position resolution. Solid = episodes "
        "0-249 (original landmark+reward placement), dashed = episodes 250-499 (after the whole value structure "
        "shifted forward 1 ring-state at episode 250).",
        color=INK_MUTED, fontsize=7.5,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    out_path = os.path.join(LOOP_DIR, out_name)
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    plot_pair("baseline", "rm",
               "Loop-position accuracy, pre vs post landmark/reward shift: baseline vs reward_modulated",
               "non_lc_loop_cycle.png")
    plot_pair("baseline_lc", "rm_lc",
               "Loop-position accuracy, pre vs post landmark/reward shift: baseline_lc vs reward_modulated_lc",
               "lc_loop_cycle.png")
    plot_pair("baseline_noshuffle", "rm_noshuffle",
               "Loop-position accuracy, pre vs post landmark/reward shift: baseline_noshuffle vs reward_modulated_noshuffle",
               "non_lc_noshuffle_loop_cycle.png")
    plot_pair("baseline_lc_noshuffle", "rm_lc_noshuffle",
               "Loop-position accuracy, pre vs post landmark/reward shift: baseline_lc_noshuffle vs reward_modulated_lc_noshuffle",
               "lc_noshuffle_loop_cycle.png")


if __name__ == "__main__":
    main()
