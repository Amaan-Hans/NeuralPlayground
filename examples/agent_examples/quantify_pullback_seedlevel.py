"""plot_pullback_heatmap.py's grid is built by averaging k{k}_activity
across all 5 seeds BEFORE row-normalizing (condition_series groups by
episode and takes the mean over seeds) - so a smooth "diagonal pullback
band" in the pooled image could in principle be an averaging artifact: 5
seeds each jumping to their settled level at a DIFFERENT, possibly random
episode would pool into something that looks like a gradual lag even if no
individual seed shows one.

This checks that directly: for EACH seed separately (no pooling), build the
same row-normalized 10 (k) x 20 (episode) grid plot_pullback_heatmap.py
plots, then for each row k find its post-shift "onset" - the (interpolated)
episode index within the 10 post-shift episodes where that row's own
globally-normalized activity first crosses 0.5. If the heatmap's apparent
pullback (activity reaching its high value LATER for higher k, i.e. further
from reward) is real at the single-seed level, onset should correlate
POSITIVELY with k within every individual seed, not just in the pooled
average. Reports the Pearson r (and Spearman rho) of k vs onset per
(condition, seed), plus the mean r and how many of the seeds agree in sign
- the real test of whether this is a per-seed phenomenon or a pooling
illusion.

Rows that never cross 0.5 during the post-shift window (their own global
max fell in the pre-shift segment and post-shift never recovers past the
midpoint) are dropped from that seed's correlation and counted separately -
a seed with many dropped rows itself argues against a clean single-seed
pullback.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop\overrepresentation_experiment"
EARLY_WINDOW = 10
K_ROWS = 10
CONDITIONS = ["baseline", "rm", "baseline_lc", "rm_lc",
              "baseline_noshuffle", "rm_noshuffle", "baseline_lc_noshuffle", "rm_lc_noshuffle"]


def build_grid_single_seed(ep_df, condition, seed):
    """Same shape/convention as plot_pullback_heatmap.py's build_grid, but
    from one seed's own rows only - no averaging across seeds."""
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
    cold-start transient (first-ever exposure, before any Hebbian memory is
    written) dominating the row's scale and making a post-shift crossing
    structurally impossible regardless of the actual post-shift dynamics."""
    post = grid[:, EARLY_WINDOW:]
    row_min = np.nanmin(post, axis=1, keepdims=True)
    row_max = np.nanmax(post, axis=1, keepdims=True)
    span = np.where(row_max - row_min > 1e-12, row_max - row_min, 1.0)
    return (post - row_min) / span


def onset_episode(y_post):
    """First (interpolated) index in y_post (10 values, post-shift segment
    of the GLOBALLY row-normalized series) where the value crosses 0.5.
    Returns np.nan if it never crosses (stays >=0.5 throughout, i.e. index
    0 already past half - still a valid onset of 0 - or never reaches 0.5
    at all, which IS the "never recovers" case worth dropping)."""
    above = y_post >= 0.5
    if above[0]:
        return 0.0
    idx = np.argmax(above)  # first True; 0 if none are True
    if not above[idx]:
        return np.nan  # never crosses
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
            # Global: row normalized over the full pre+post range (matches
            # what the heatmap image itself shows) - but a pre-shift
            # episode-0 cold-start spike can dominate this scale (see
            # module docstring) and make a post-shift crossing impossible
            # regardless of real dynamics.
            norm_global = normalize_rows(grid)[:, EARLY_WINDOW:]
            rows.append(_correlate_onsets(norm_global, "global", cond, seed))
            # Post-local: row normalized using ONLY the post-shift segment's
            # own min/max - immune to the pre-shift cold-start spike, a
            # cleaner test of within-post-segment lag structure vs k.
            norm_post_local = normalize_post_segment(grid)
            rows.append(_correlate_onsets(norm_post_local, "post_local", cond, seed))
    return pd.DataFrame(rows)


def main():
    ep_df = pd.read_csv(os.path.join(OUT_DIR, "pullback_overtime_episodes.csv"))
    seeds = sorted(ep_df["seed"].unique())
    per_seed = per_seed_correlations(ep_df, CONDITIONS, seeds)
    per_seed.to_csv(os.path.join(OUT_DIR, "pullback_seedlevel_correlations.csv"), index=False)
    print(per_seed.to_string(index=False))

    summary = per_seed.groupby(["condition", "norm"]).agg(
        mean_pearson_r=("pearson_r", "mean"),
        sem_pearson_r=("pearson_r", "sem"),
        n_seeds_positive_r=("pearson_r", lambda s: int((s > 0).sum())),
        n_seeds_total=("pearson_r", "count"),
        mean_spearman_rho=("spearman_rho", "mean"),
    ).reset_index()
    summary["condition"] = pd.Categorical(summary["condition"], categories=CONDITIONS, ordered=True)
    summary = summary.sort_values(["condition", "norm"])
    summary.to_csv(os.path.join(OUT_DIR, "pullback_seedlevel_summary.csv"), index=False)
    print("\n=== per-condition summary (across seeds) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved pullback_seedlevel_correlations.csv and pullback_seedlevel_summary.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
