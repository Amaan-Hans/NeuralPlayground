"""Plot a handful of individual place-cell tuning curves around the
20-state ring, pre- vs post-shift, with the real landmark/control/reward
layout overlaid - a quick visual sanity check that (a) the ring geometry
matches the intended design and (b) place cells show sensible spatial
tuning along it (bumps, not noise). Uses the already-computed
*_place_cells.csv from the 40-run batch (no re-running needed). Saved to
the user's Downloads folder.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOOP_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop"
DOWNLOADS = r"C:\Users\hansl\Downloads"
COND = "rm"
SEED = 42
N_RING_STATES = 20
N_CELLS_TO_PLOT = 6

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GOLD = "#eda100"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

CAT_COLORS = {"landmark": BLUE, "control": ORANGE, "reward": GOLD, "filler": "#cfcdc4"}


def category_strip(df_half):
    """is_landmark/is_control_landmark/is_reward -> one category per ring_pos."""
    cats = []
    for pos in range(N_RING_STATES):
        row = df_half[df_half["ring_pos"] == pos].iloc[0]
        if row["is_reward"]:
            cats.append("reward")
        elif row["is_landmark"]:
            cats.append("landmark")
        elif row["is_control_landmark"]:
            cats.append("control")
        else:
            cats.append("filler")
    return cats


def draw_category_strip(ax, cats, y_frac=1.03):
    for pos, cat in enumerate(cats):
        ax.axvspan(pos - 0.5, pos + 0.5, color=CAT_COLORS[cat], alpha=0.12, lw=0)


def main():
    df = pd.read_csv(os.path.join(LOOP_DIR, f"{COND}_seed{SEED}_place_cells.csv"))
    df0 = df[df["env_id"] == 0]
    df_pre = df0[df0["half"] == "pre"].set_index("ring_pos").reindex(range(N_RING_STATES))
    df_post = df0[df0["half"] == "post"].set_index("ring_pos").reindex(range(N_RING_STATES))

    cats_pre = category_strip(df0[df0["half"] == "pre"])
    cats_post = category_strip(df0[df0["half"] == "post"])

    p_cols = [c for c in df.columns if c.startswith("p_")]
    # Pick the cells with the largest pre-shift peak-to-trough range - the
    # ones most likely to show a clean, visually-legible bump rather than
    # a flat/noisy trace.
    ranges = (df_pre[p_cols].max() - df_pre[p_cols].min()).sort_values(ascending=False)
    chosen = list(ranges.index[:N_CELLS_TO_PLOT])

    fig, axes = plt.subplots(N_CELLS_TO_PLOT, 1, figsize=(10, 2.0 * N_CELLS_TO_PLOT), facecolor=PAGE, sharex=True)
    for ax, cell in zip(axes, chosen):
        draw_category_strip(ax, cats_pre)
        ax.plot(range(N_RING_STATES), df_pre[cell], color=BLUE, linewidth=1.8, marker="o", markersize=3,
                label="pre-shift")
        ax.plot(range(N_RING_STATES), df_post[cell], color=ORANGE, linewidth=1.8, linestyle="--", marker="o",
                markersize=3, label="post-shift")
        ax.set_facecolor(SURFACE)
        ax.set_ylabel(cell.replace("p_", ""), color=INK_SECONDARY, fontsize=8.5)
        ax.set_xlim(-0.5, N_RING_STATES - 0.5)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c3c2b7")
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK_SECONDARY, ncol=2)
    axes[-1].set_xlabel("ring position (state index, 0-19)", color=INK_SECONDARY, fontsize=9.5)
    axes[-1].set_xticks(range(N_RING_STATES))

    legend_items = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.3, label=cat)
                    for cat, c in CAT_COLORS.items()]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"Place-cell tuning along the ring ({COND}, seed {SEED}) - shaded bands = pre-shift layout",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.965,
              "6 cells with the largest pre-shift peak-to-trough range, picked for visual legibility. Bumps should "
              "sit near landmark/reward positions (blue/gold bands) and track the shift forward by 1 ring-state "
              "post-shift (dashed) if the loop geometry and the shift mechanic are both working correctly.",
              ha="center", va="top", color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=[0, 0.035, 1, 0.94])

    out_path = os.path.join(DOWNLOADS, "loop_place_fields.png")
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"Saved {out_path}")

    print("\nPre-shift category strip (ring_pos 0-19):", cats_pre)
    print("Post-shift category strip (ring_pos 0-19):", cats_post)


if __name__ == "__main__":
    main()
