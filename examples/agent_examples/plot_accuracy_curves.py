"""Plot the 5-seed, 500-episode accuracy curves from probe_accuracy_curve.py
- one pair-plot for the non-LC conditions (baseline vs rm, plus the
rotation-OFF baseline_noshuffle/rm_noshuffle as flat reference lines) and
one for the LC conditions (baseline_lc vs rm_lc - no noshuffle counterpart
exists for LC), saved as PNGs to a dedicated Downloads folder.

Rotation-enabled conditions are pooled two ways at once: across all ~20
within-seed rotation cycles AND across all 5 seeds, onto a single 0-24
"episodes since last reshuffle" axis (mean +/- SEM over every seed x cycle
combination, ~100 samples per x-position). Rotation-OFF conditions have no
cycle structure to pool by, so they're shown as a flat mean +/- SEM band
across all 5 seeds x all 500 episodes instead - the honest comparison is
"where does the no-rotation network sit overall" vs "how does the
rotation-enabled network's accuracy move within a cycle."
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROBE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accuracy_curve_results")
OUT_DIR = r"C:\Users\hansl\Downloads\tem_accuracy_curves_seed42"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"

SEEDS = [42, 7, 1, 2, 3]
ROTATION_INTERVAL = 25


def style_axes(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE_AXIS)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def load_all_seeds(label):
    frames = []
    for seed in SEEDS:
        path = os.path.join(PROBE_DIR, f"{label}_seed{seed}_accuracycurve.csv")
        if not os.path.exists(path):
            print(f"missing: {path}")
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _pool_by_cycle_position(df, col):
    """Align every 25-episode rotation cycle (within every seed) to a
    common 0..24 x-axis (position_in_cycle = episode % ROTATION_INTERVAL -
    position 0 is always the first episode right after a reshuffle) and
    average across all seed x cycle combinations at each position."""
    sub = df.dropna(subset=[col]).copy()
    sub["position_in_cycle"] = sub["episode"] % ROTATION_INTERVAL
    return sub.groupby("position_in_cycle")[col].agg(["mean", "std", "count"])


def _plot_pooled_line(ax, df, col, color, label):
    g = _pool_by_cycle_position(df, col)
    ax.plot(g.index, g["mean"], color=color, linewidth=1.8, alpha=0.9, label=label, marker="o", markersize=3)
    sem = g["std"] / np.sqrt(g["count"].clip(lower=1))
    ax.fill_between(g.index, g["mean"] - sem, g["mean"] + sem, color=color, alpha=0.15, linewidth=0)


def _plot_flat_reference(ax, df, col, color, label, x_range):
    sub = df.dropna(subset=[col])
    mean = sub[col].mean()
    sem = sub[col].std() / np.sqrt(len(sub))
    ax.axhline(mean, color=color, linewidth=1.6, linestyle="--", alpha=0.85, label=label, zorder=2)
    ax.fill_between(x_range, mean - sem, mean + sem, color=color, alpha=0.12, linewidth=0, zorder=1)


def plot_pair(baseline_label, rm_label, campaign_title, out_name, noshuffle_labels=None):
    df_bl = load_all_seeds(baseline_label)
    df_rm = load_all_seeds(rm_label)
    df_bl_ns = load_all_seeds(noshuffle_labels[0]) if noshuffle_labels else None
    df_rm_ns = load_all_seeds(noshuffle_labels[1]) if noshuffle_labels else None

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=PAGE, sharey=True)
    panels = [
        ("x_gt_landmark", "Value landmarks (object-prediction accuracy)"),
        ("x_gt_control", "Control landmarks (object-prediction accuracy)"),
        ("x_gt_filler", "Filler/remainder states (object-prediction accuracy)"),
    ]
    x_range = np.array([0, ROTATION_INTERVAL - 1])
    for ax, (col, subtitle) in zip(axes, panels):
        _plot_pooled_line(ax, df_bl, col, BLUE, baseline_label)
        _plot_pooled_line(ax, df_rm, col, ORANGE, rm_label)
        if df_bl_ns is not None:
            _plot_flat_reference(ax, df_bl_ns, col, AQUA, noshuffle_labels[0] + " (no rotation)", x_range)
            _plot_flat_reference(ax, df_rm_ns, col, YELLOW, noshuffle_labels[1] + " (no rotation)", x_range)
        ax.axvline(0, color=GRIDLINE, linewidth=0.8, zorder=1, linestyle="--")
        ax.text(0.3, 0.02, "reshuffle\njust happened", transform=ax.get_xaxis_transform(),
                color=INK_MUTED, fontsize=7, va="bottom")
        ax.set_xlabel("episode since last control-landmark reshuffle", color=INK_SECONDARY, fontsize=9)
        ax.set_xlim(0, ROTATION_INTERVAL - 1)
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK_SECONDARY)
        style_axes(ax, subtitle)
    axes[0].set_ylabel("x_gt accuracy (object identity)", color=INK_SECONDARY, fontsize=9)

    fig.suptitle(campaign_title, color=INK_PRIMARY, fontsize=13, fontweight="bold")
    fig.text(
        0.01, 0.01,
        f"5 seeds, frozen weights - rotation-enabled conditions: all seeds x ~{500 // ROTATION_INTERVAL} cycles "
        f"pooled onto one 0-{ROTATION_INTERVAL - 1} within-cycle axis (mean +/- SEM); "
        f"rotation-off conditions (dashed): flat mean +/- SEM across all 5 seeds x 500 episodes, no cycle structure",
        color=INK_MUTED, fontsize=7.5,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_pair(
        "baseline", "rm",
        "Accuracy by object category, pooled across 5 seeds: baseline vs reward_modulated (+ rotation-off reference)",
        "non_lc_accuracy_curve.png",
        noshuffle_labels=("baseline_noshuffle", "rm_noshuffle"),
    )
    plot_pair(
        "baseline_lc", "rm_lc",
        "Accuracy by object category, pooled across 5 seeds: baseline_lc vs reward_modulated_lc",
        "lc_accuracy_curve.png",
    )


if __name__ == "__main__":
    main()
