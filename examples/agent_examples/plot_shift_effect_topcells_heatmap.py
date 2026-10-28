"""Heatmap version of quantify_shift_effect_topcells.py's result: rows =
condition, columns = k (0-4, reward-approach position), color = % change in
top-10-cell activity, post vs pre shift. Diverging colormap centered on 0
so positive (red, activity went up post-shift) vs negative (blue, went
down) is immediate at a glance, with the mean % change annotated in each
cell. Saved into Thesis/quantitative/loop/overrepresentation_experiment/.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop\overrepresentation_experiment"
CONDITIONS = ["baseline", "rm", "baseline_lc", "rm_lc",
              "baseline_noshuffle", "rm_noshuffle", "baseline_lc_noshuffle", "rm_lc_noshuffle"]
K_RANGE = range(5)
K_LABELS = ["k=0\n(reward)", "k=1\n(landmark 4)", "k=2\n(control/filler)", "k=3\n(landmark 3)", "k=4\n(control/filler)"]

PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def main():
    summary = pd.read_csv(os.path.join(OUT_DIR, "shift_effect_topcells_summary.csv"))

    pct_grid = np.full((len(CONDITIONS), len(K_RANGE)), np.nan)
    sem_grid = np.full((len(CONDITIONS), len(K_RANGE)), np.nan)
    for i, cond in enumerate(CONDITIONS):
        for k in K_RANGE:
            row = summary[(summary["condition"] == cond) & (summary["k"] == k)]
            if len(row):
                pct_grid[i, k] = row["pct_change_mean"].values[0]
                sem_grid[i, k] = row["pct_change_sem"].values[0]

    vmax = np.nanmax(np.abs(pct_grid))
    fig, ax = plt.subplots(figsize=(8, 7.5), facecolor=PAGE)
    im = ax.imshow(pct_grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(K_RANGE)))
    ax.set_xticklabels(K_LABELS, fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels(CONDITIONS, fontsize=10, color=INK_SECONDARY)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(len(CONDITIONS)):
        for k in K_RANGE:
            val, sem = pct_grid[i, k], sem_grid[i, k]
            text_color = "white" if abs(val) > vmax * 0.55 else INK_PRIMARY
            ax.text(k, i, f"{val:+.0f}%\n(\u00b1{sem:.0f})", ha="center", va="center",
                    color=text_color, fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("% change in top-cell activity, post vs pre shift", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)

    ax.set_title("Effect of the shift (top-10-cell activity), all conditions",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold", pad=14)
    fig.text(0.5, 0.015,
              "Mean % change (\u00b1 SEM, 5 seeds) in the top 10 most-active cells' firing rate at each position, "
              "real 250-episode settled pre vs post. Red = activity rose after the shift, blue = fell.",
              ha="center", va="bottom", color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = os.path.join(OUT_DIR, "shift_effect_topcells_heatmap.png")
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
