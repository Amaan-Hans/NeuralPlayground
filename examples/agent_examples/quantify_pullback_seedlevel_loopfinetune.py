"""Same onset-vs-k correlation method as quantify_pullback_seedlevel.py /
_decoytile.py, pointed at the loop-finetune experiment
(Thesis2/quantitative/loop_finetune/overrepresentation_experiment/). Only
seed 42 exists here, so there's no averaging-across-seeds question to
settle - this just reports that single run's own per-condition Pearson r /
Spearman rho, for the 3-way comparison against the other two campaigns'
per-seed numbers.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis2\quantitative\loop_finetune\overrepresentation_experiment"
EARLY_WINDOW = 10
K_ROWS = 10
CONDITIONS = ["baseline", "rm", "baseline_lc", "rm_lc",
              "baseline_noshuffle", "rm_noshuffle", "baseline_lc_noshuffle", "rm_lc_noshuffle"]


def build_grid_single_seed(ep_df, condition, seed):
    sub = ep_df[(ep_df["condition"] == condition) & (ep_df["seed"] == seed)]
    grid = np.full((K_ROWS, 2 * EARLY_WINDOW), np.nan)
    for k in range(K_ROWS):
        pre = sub[sub["era"] == "early_pre"].set_index("episode")[f"k{k}_activity"].reindex(range(EARLY_WINDOW))
        post = sub[sub["era"] == "early_post"].set_index("episode")[f"k{k}_activity"].reindex(range(EARLY_WINDOW))
        grid[k] = np.concatenate([pre.values, post.values])
    return grid


def normalize_rows(grid):
    row_min = np.nanmin(grid, axis=1, keepdims=True)
    row_max = np.nanmax(grid, axis=1, keepdims=True)
    span = np.where(row_max - row_min > 1e-12, row_max - row_min, 1.0)
    return (grid - row_min) / span


def normalize_post_segment(grid):
    """Normalize each row using ONLY its own post-shift (columns 10-19)
    min/max, not the full pre+post range - avoids the pre-shift episode-0
    cold-start transient dominating the row's scale."""
    post = grid[:, EARLY_WINDOW:]
    row_min = np.nanmin(post, axis=1, keepdims=True)
    row_max = np.nanmax(post, axis=1, keepdims=True)
    span = np.where(row_max - row_min > 1e-12, row_max - row_min, 1.0)
    return (post - row_min) / span


def onset_episode(y_post):
    above = y_post >= 0.5
    if above[0]:
        return 0.0
    idx = np.argmax(above)
    if not above[idx]:
        return np.nan
    y0, y1 = y_post[idx - 1], y_post[idx]
    if y1 == y0:
        return float(idx)
    frac = (0.5 - y0) / (y1 - y0)
    return float(idx - 1) + frac


def _correlate_onsets(norm_post, label, cond, seed):
    ks, onsets = [], []
    n_dropped = 0
    for k in range(K_ROWS):
        onset = onset_episode(norm_post[k])
        if np.isnan(onset):
            n_dropped += 1
            continue
        ks.append(k)
        onsets.append(onset)
    if len(ks) >= 3:
        r, p = stats.pearsonr(ks, onsets)
        rho, p_rho = stats.spearmanr(ks, onsets)
    else:
        r, p, rho, p_rho = np.nan, np.nan, np.nan, np.nan
    return {
        "condition": cond, "seed": seed, "norm": label, "n_rows_used": len(ks), "n_rows_dropped": n_dropped,
        "pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": p_rho,
    }


def per_seed_correlations(ep_df, conditions, seeds):
    rows = []
    for cond in conditions:
        for seed in seeds:
            grid = build_grid_single_seed(ep_df, cond, seed)
            norm_global = normalize_rows(grid)[:, EARLY_WINDOW:]
            rows.append(_correlate_onsets(norm_global, "global", cond, seed))
            norm_post_local = normalize_post_segment(grid)
            rows.append(_correlate_onsets(norm_post_local, "post_local", cond, seed))
    return pd.DataFrame(rows)


def main():
    ep_df = pd.read_csv(os.path.join(OUT_DIR, "pullback_overtime_episodes.csv"))
    seeds = sorted(ep_df["seed"].unique())
    per_seed = per_seed_correlations(ep_df, CONDITIONS, seeds)
    per_seed.to_csv(os.path.join(OUT_DIR, "pullback_seedlevel_correlations.csv"), index=False)
    print(per_seed.to_string(index=False))
    print(f"\nSaved pullback_seedlevel_correlations.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
