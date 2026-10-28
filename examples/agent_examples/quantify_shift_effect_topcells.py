"""Same question as quantify_shift_effect.py (effect of the shift on
activity at k=0-4, all 8 conditions) but using only the MOST ACTIVE place
cells at each position instead of averaging over the whole ~440-cell
population - averaging over everything dilutes the signal with cells that
aren't really tuned to that location at all. For each position, we take the
top TOP_N cells by firing rate there (independently in the pre half and
the post half, since remapping across the shift is expected/allowed - this
asks "how strong is the best available representation of this location",
not "did these exact cells keep firing here").

Reads the real per-cell settled rate maps directly from the main batch's
*_place_cells.csv (not the summed/pooled pullback_overtime files), so this
uses the true 250-episode-averaged per-cell rates, all 8 conditions x 5
seeds.

Saves shift_effect_topcells_k0_4.csv (full per-seed table) and
shift_effect_topcells_summary.csv (condition x k means/SEMs) to
Thesis/quantitative/loop/overrepresentation_experiment/, plus a grouped bar
chart matching quantify_shift_effect.py's layout for direct comparison.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOOP_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop"
OUT_DIR = os.path.join(LOOP_DIR, "overrepresentation_experiment")
CONDITIONS = ["baseline", "rm", "baseline_lc", "rm_lc",
              "baseline_noshuffle", "rm_noshuffle", "baseline_lc_noshuffle", "rm_lc_noshuffle"]
SEEDS = [42, 7, 1, 2, 3]
K_RANGE = range(5)
TOP_N = 30  # most active cells per position, per half (was 10 - too few, noisy per-seed)

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


def load_rate_map(cond, seed, half):
    df = pd.read_csv(os.path.join(LOOP_DIR, f"{cond}_seed{seed}_place_cells.csv"))
    dfh = df[df["half"] == half]
    p_cols = [c for c in dfh.columns if c.startswith("p_")]
    grouped = dfh.groupby("ring_pos")[p_cols].mean()
    reward_pos = dfh.groupby("ring_pos").first()["is_reward"]
    reward_pos = reward_pos[reward_pos == 1].index[0]
    n_ring = dfh["ring_pos"].nunique()
    return grouped, int(reward_pos), n_ring


def top_cell_mean(rates, pos, n=TOP_N):
    row = rates.loc[pos]
    return row.nlargest(n).mean()


def main():
    rows = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            pre_rates, reward_pre, n_ring = load_rate_map(cond, seed, "pre")
            post_rates, reward_post, _ = load_rate_map(cond, seed, "post")
            for k in K_RANGE:
                pos_pre = (reward_pre - k) % n_ring
                pos_post = (reward_post - k) % n_ring
                v_pre = top_cell_mean(pre_rates, pos_pre)
                v_post = top_cell_mean(post_rates, pos_post)
                rows.append({
                    "condition": cond, "seed": seed, "k": k,
                    "pre": v_pre, "post": v_post,
                    "delta": v_post - v_pre,
                    "pct_change": (v_post - v_pre) / v_pre * 100 if v_pre else np.nan,
                })

    full = pd.DataFrame(rows)
    full.to_csv(os.path.join(OUT_DIR, "shift_effect_topcells_k0_4.csv"), index=False)

    summary = full.groupby(["condition", "k"]).agg(
        pre_mean=("pre", "mean"), post_mean=("post", "mean"),
        delta_mean=("delta", "mean"), delta_sem=("delta", "sem"),
        pct_change_mean=("pct_change", "mean"), pct_change_sem=("pct_change", "sem"),
    ).reset_index()
    summary["condition"] = pd.Categorical(summary["condition"], categories=CONDITIONS, ordered=True)
    summary = summary.sort_values(["condition", "k"])
    summary.to_csv(os.path.join(OUT_DIR, "shift_effect_topcells_summary.csv"), index=False)
    print(summary.to_string(index=False))

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
    ax.set_ylabel("% change in top-cell activity, post vs pre shift", color=INK_SECONDARY, fontsize=10)
    ax.set_title(f"Effect of the shift using the top {TOP_N} most-active place cells per position, all conditions",
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
              f"From the REAL 250-episode settled pre/post rate maps (5 seeds, mean +/- SEM). For each position, "
              f"only the top {TOP_N} cells by firing rate THERE are averaged - selected independently in pre and "
              "post (remapping allowed) - instead of the whole ~440-cell population. "
              "k=0 reward (gold), k=1/3 actual landmarks (blue), k=2/4 control/filler gaps (grey).",
              ha="center", va="bottom", color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = os.path.join(OUT_DIR, "shift_effect_topcells_k0_4.png")
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
