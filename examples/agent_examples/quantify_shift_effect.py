"""Quantifies the effect of the episode-shift on place-cell activity at the
5 ring positions closest to reward: k=0 (reward itself) through k=4
(landmark 3, 3 ring-states before reward, plus the 2 control/filler gaps
k=2 and k=4 in between) - "backward" from reward along the approach
sequence.

Uses the REAL settled (250-episode) pre vs post activity from
pullback_overtime_baseline.csv - not the short 10-episode probe, which is
dominated by the episode-0 cold-start transient (see
plot_pullback_heatmap.py's findings) and too noisy for a clean effect size.
The settled baseline is the actual long-run answer to "what does the shift
do to activity at this position", per condition, per seed.

For each condition x k, computes the per-seed delta (post - pre) and
percent change ((post-pre)/pre*100), then the mean +/- SEM across the 5
seeds. Saves shift_effect_k0_4.csv (full per-seed table) and
shift_effect_summary.csv (condition x k means/SEMs) to
Thesis/quantitative/loop/overrepresentation_experiment/, plus a grouped bar
chart (shift_effect_k0_4.png) of percent change, one group of bars (k=0..4)
per condition.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop\overrepresentation_experiment"
CONDITIONS = ["baseline", "rm", "baseline_lc", "rm_lc",
              "baseline_noshuffle", "rm_noshuffle", "baseline_lc_noshuffle", "rm_lc_noshuffle"]
K_RANGE = range(5)  # k=0..4

GOLD = "#eda100"
BLUE = "#2a78d6"
GREY = "#898781"
PAGE = "#f9f9f7"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"

K_LABELS = {0: "k=0 (reward)", 1: "k=1 (landmark 4)", 2: "k=2 (control/filler)",
            3: "k=3 (landmark 3)", 4: "k=4 (control/filler)"}
K_COLORS = {0: GOLD, 1: BLUE, 2: GREY, 3: BLUE, 4: GREY}


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "pullback_overtime_baseline.csv"))

    pre = df[df["half"] == "pre"].set_index(["condition", "seed"])
    post = df[df["half"] == "post"].set_index(["condition", "seed"])

    rows = []
    for cond in CONDITIONS:
        for seed in sorted(df["seed"].unique()):
            for k in K_RANGE:
                col = f"k{k}_activity"
                v_pre = pre.loc[(cond, seed), col]
                v_post = post.loc[(cond, seed), col]
                rows.append({
                    "condition": cond, "seed": seed, "k": k,
                    "pre": v_pre, "post": v_post,
                    "delta": v_post - v_pre,
                    "pct_change": (v_post - v_pre) / v_pre * 100 if v_pre else np.nan,
                })
    full = pd.DataFrame(rows)
    full.to_csv(os.path.join(OUT_DIR, "shift_effect_k0_4.csv"), index=False)

    summary = full.groupby(["condition", "k"]).agg(
        pre_mean=("pre", "mean"), post_mean=("post", "mean"),
        delta_mean=("delta", "mean"), delta_sem=("delta", "sem"),
        pct_change_mean=("pct_change", "mean"), pct_change_sem=("pct_change", "sem"),
    ).reset_index()
    summary["condition"] = pd.Categorical(summary["condition"], categories=CONDITIONS, ordered=True)
    summary = summary.sort_values(["condition", "k"])
    summary.to_csv(os.path.join(OUT_DIR, "shift_effect_summary.csv"), index=False)
    print(summary.to_string(index=False))

    # Grouped bar chart: percent change, one group of 5 bars (k=0..4) per condition.
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=PAGE)
    ax.set_facecolor(SURFACE)
    n_k = len(K_RANGE)
    width = 0.15
    x = np.arange(len(CONDITIONS))
    for i, k in enumerate(K_RANGE):
        sub = summary[summary["k"] == k].set_index("condition").reindex(CONDITIONS)
        offset = (i - (n_k - 1) / 2) * width
        ax.bar(x + offset, sub["pct_change_mean"], width, yerr=sub["pct_change_sem"],
               color=K_COLORS[k], alpha=0.55 + 0.1 * (k % 2), label=K_LABELS[k],
               error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 2})
    ax.axhline(0, color=BASELINE_AXIS, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, rotation=20, ha="right", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("% change in settled activity, post vs pre shift", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Effect of the shift on place-cell activity at k=0-4 (reward-approach positions), all conditions",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE_AXIS)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK_SECONDARY, ncol=5)
    fig.text(0.5, 0.01,
              "From the REAL 250-episode settled pre/post activity (5 seeds, mean +/- SEM), not the short probe. "
              "k=0 reward (gold), k=1/3 actual landmarks (blue), k=2/4 control/filler gaps (grey).",
              ha="center", va="bottom", color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = os.path.join(OUT_DIR, "shift_effect_k0_4.png")
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
