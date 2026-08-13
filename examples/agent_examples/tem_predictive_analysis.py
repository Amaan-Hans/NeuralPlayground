"""Post-hoc analysis: do reward-condition place cells become predictive of reward?

Run this after both training runs have finished.

Produces three comparisons across episode checkpoints:

1. Population activity map
   Mean firing across all place cells per grid state, plotted on the 2D arena.
   Reward condition: activity focus should spread backward from reward over training.

2. Value-cell correlation (reward condition only)
   Pearson correlation between each state's mean place cell activity and V(s).
   Grows over episodes if cells become predictive of future reward.

3. Peak firing distance from reward
   For each cell, find the state with peak firing and compute its Euclidean
   distance to the reward location.  Compares reward vs baseline to show whether
   reward condition cells develop peaks further from the reward site (backward
   shift) while baseline cells stay random.

(Also produces grid_scores.png, proximal_cell_count.png, and
reward_zone_enrichment.png / reward_zone_field_distance_hist.png - the last
two need tem_probe_eval_multienv.py to have been run first, since they read
its saved probe_<condition>_rates.npz/_meta.pkl from this same output
directory, not from the per-checkpoint training plots.)

Usage
-----
    cd examples/agent_examples
    python tem_predictive_analysis.py
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from neuralplayground.comparison import GridScorer

# ── Configuration ──────────────────────────────────────────────────────────────
TEST_MODE = os.environ.get("TEM_TEST_MODE", "0") == "1"  # True = read from results_sim_test/

_results_folder = "results_sim_test" if TEST_MODE else "results_sim"
RESULTS_ROOT  = os.path.join(os.getcwd(), _results_folder)
BASELINE_DIR  = os.path.join(RESULTS_ROOT, "baseline",         "plots")
REWARD_DIR    = os.path.join(RESULTS_ROOT, "reward_modulated", "plots")
OUT_DIR       = os.path.join(RESULTS_ROOT, "predictive_analysis")
PROBE_DIR     = os.path.join(OUT_DIR, "probe")

# Checkpoints after this episode are treated as "loop phase" for proximal cell count.
# Set to 0 in TEST_MODE so all checkpoints are included.
LOOP_START_EPISODE  = 0 if TEST_MODE else 1000
# Proximity threshold for counting reward-proximal cells (grid units)
PROXIMAL_THRESHOLD  = 2.0
os.makedirs(OUT_DIR, exist_ok=True)

REWARD_LOCATION = np.array([3.0, 3.0])
ROOM_W, ROOM_D  = 10, 10          # env 0: [-5,5] x [-5,5], state_density=1
STATE_DENSITY   = 1
N_STATES        = ROOM_W * ROOM_D  # 100

# Grid cell centres (matches DiscreteObjectEnvironment xy_combination for env 0)
_x = np.linspace(-ROOM_W/2 + 0.5/STATE_DENSITY, ROOM_W/2 - 0.5/STATE_DENSITY, ROOM_W)
_y = np.linspace(-ROOM_D/2 + 0.5/STATE_DENSITY, ROOM_D/2 - 0.5/STATE_DENSITY, ROOM_D)
XY = np.stack(np.meshgrid(_x, _y), axis=-1)          # (ROOM_D, ROOM_W, 2)
XY_FLAT = XY.reshape(-1, 2)                           # (N_STATES, 2)  row = state id

# Distance from each state to the reward location
DIST_TO_REWARD = np.linalg.norm(XY_FLAT - REWARD_LOCATION, axis=1)  # (N_STATES,)

# ── Reward-zone field enrichment configuration ─────────────────────────────────
# Zone radius matches PROXIMAL_THRESHOLD's convention (same "how close counts as
# near-reward" judgment call used by the existing proximal-cell-count metric).
ZONE_RADIUS         = 2.0
FIELD_THRESH_FRAC   = 0.5   # field = contiguous region >= this fraction of a cell's own peak
MIN_FIELD_SIZE      = 2     # states; discards single-pixel noise "fields"
SELECTIVITY_THRESH  = 2.5   # peak/mean ratio a cell's rate map must clear to count as place-like
MIN_AMPLITUDE       = 0.02  # peak - min must exceed this; excludes near-flat/dead cells
N_SHUFFLES          = 2000
SHUFFLE_SEED        = 0
# ──────────────────────────────────────────────────────────────────────────────


def _episode_dirs(plots_dir):
    """Return sorted list of (episode_int, path) for every episode_N folder."""
    entries = []
    if not os.path.isdir(plots_dir):
        return entries
    for name in os.listdir(plots_dir):
        if name.startswith("episode_"):
            try:
                ep = int(name.split("_")[1])
                entries.append((ep, os.path.join(plots_dir, name)))
            except ValueError:
                pass
    return sorted(entries)


def _load_p_rates(ep_path):
    """Load p_rates.npy -> (N_STATES, n_cells) or None."""
    fpath = os.path.join(ep_path, "p_rates.npy")
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)          # (N_STATES, total_place_cells)


def _load_v_table(ep_path):
    """Load v_table.npy -> (N_STATES,) or None."""
    fpath = os.path.join(ep_path, "v_table.npy")
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)


# ── Analysis 1: Population activity map ───────────────────────────────────────

def plot_population_activity_maps():
    """For each condition, tile population-activity heatmaps across episodes."""
    for label, plots_dir in [("baseline", BASELINE_DIR), ("reward_modulated", REWARD_DIR)]:
        eps = _episode_dirs(plots_dir)
        if not eps:
            print(f"No episode folders found in {plots_dir}")
            continue

        n = len(eps)
        ncols = min(5, n)
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(nrows, ncols,
                                figsize=(ncols * 3, nrows * 3 + 0.5))
        axs = np.array(axs).reshape(nrows, ncols)
        fig.suptitle(f"Population activity map — {label}", fontsize=13)

        vmax_global = 0
        maps = []
        for ep, ep_path in eps:
            p = _load_p_rates(ep_path)
            if p is None:
                maps.append(None)
                continue
            act = p.mean(axis=1)                        # (N_STATES,)
            maps.append((ep, act))
            vmax_global = max(vmax_global, act.max())

        for idx, item in enumerate(maps):
            row, col = divmod(idx, ncols)
            ax = axs[row, col]
            if item is None:
                ax.axis("off")
                continue
            ep, act = item
            grid = act.reshape(ROOM_D, ROOM_W)
            im = ax.imshow(grid, origin="lower", cmap="hot",
                           vmin=0, vmax=vmax_global, aspect="auto")
            # Mark reward location
            rx = (REWARD_LOCATION[0] + ROOM_W/2) / ROOM_W * ROOM_W - 0.5
            ry = (REWARD_LOCATION[1] + ROOM_D/2) / ROOM_D * ROOM_D - 0.5
            ax.plot(rx, ry, "*", color="cyan", markersize=10)
            ax.set_title(f"ep {ep}", fontsize=8)
            ax.axis("off")

        for idx in range(len(maps), nrows * ncols):
            row, col = divmod(idx, ncols)
            axs[row, col].axis("off")

        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f"population_activity_{label}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved: {fname}")


# ── Analysis 2: Value-cell correlation over episodes ──────────────────────────

def plot_value_correlation():
    """Correlation between mean place cell activity and V(landmark), evaluated
    only at the landmark states. Only possible for reward_modulated (V table
    only saved there).

    v_table.npy is V(landmark) projected onto each landmark's unique state,
    NaN everywhere else (non-landmark states have no fixed value of their
    own — see Useful_info/experiment_changes.md). Correlating against the
    full (mostly-NaN) array makes np.corrcoef return NaN for every checkpoint
    — that's why this plot used to come out blank. Restrict to the
    non-NaN (landmark) entries instead.
    """
    eps = _episode_dirs(REWARD_DIR)
    episodes, correlations = [], []

    for ep, ep_path in eps:
        p = _load_p_rates(ep_path)
        v = _load_v_table(ep_path)
        if p is None or v is None:
            continue
        mean_act = p.mean(axis=1)                       # (N_STATES,)
        valid = ~np.isnan(v)
        if valid.sum() < 2:
            continue
        mean_act_valid, v_valid = mean_act[valid], v[valid]
        # Pearson correlation
        if mean_act_valid.std() < 1e-8 or v_valid.std() < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(mean_act_valid, v_valid)[0, 1])
        episodes.append(ep)
        correlations.append(corr)

    if not episodes:
        print("No V table data found — skipping value correlation plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(episodes, correlations, "o-", color="darkorange", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Pearson r  (mean place activity vs V, at landmark states)")
    ax.set_title("Place cell–value correlation over training\n(reward_modulated, env 0, landmark states only)")
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "value_correlation.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# ── Analysis 3: Peak firing distance from reward ──────────────────────────────

def plot_peak_distance():
    """Distribution of each cell's peak-firing distance from reward,
    for both conditions across episodes.  A backward shift shows up as
    reward_modulated cells developing peaks at LARGER distances (further back
    along the approach path) compared to baseline.
    """
    results = {}
    for label, plots_dir in [("baseline", BASELINE_DIR), ("reward_modulated", REWARD_DIR)]:
        eps = _episode_dirs(plots_dir)
        ep_list, mean_dist, median_dist = [], [], []
        for ep, ep_path in eps:
            p = _load_p_rates(ep_path)
            if p is None:
                continue
            peak_states = np.argmax(p, axis=0)          # (n_cells,)
            distances   = DIST_TO_REWARD[peak_states]   # (n_cells,)
            ep_list.append(ep)
            mean_dist.append(distances.mean())
            median_dist.append(np.median(distances))
        results[label] = (ep_list, mean_dist, median_dist)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    colours = {"baseline": "steelblue", "reward_modulated": "darkorange"}

    for ax, metric, title in zip(axes, ["mean", "median"],
                                 ["Mean", "Median"]):
        for label, (ep_list, mean_d, med_d) in results.items():
            vals = mean_d if metric == "mean" else med_d
            if ep_list:
                ax.plot(ep_list, vals, "o-", color=colours[label],
                        label=label, linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Distance from reward (grid units)")
        ax.set_title(f"{title} peak-firing distance from reward")
        ax.legend()

    fig.suptitle("Backward shift: place cell peaks vs reward location (env 0)", fontsize=12)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "peak_distance_from_reward.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


    # Also save per-episode histogram comparison at first and last checkpoint
    for label, plots_dir in [("baseline", BASELINE_DIR), ("reward_modulated", REWARD_DIR)]:
        eps = _episode_dirs(plots_dir)
        checkpoints = [eps[0], eps[-1]] if len(eps) >= 2 else eps
        if not checkpoints:
            continue
        fig, axs = plt.subplots(1, len(checkpoints),
                                figsize=(5 * len(checkpoints), 4), sharey=False)
        if len(checkpoints) == 1:
            axs = [axs]
        for ax, (ep, ep_path) in zip(axs, checkpoints):
            p = _load_p_rates(ep_path)
            if p is None:
                ax.axis("off")
                continue
            peak_states = np.argmax(p, axis=0)
            distances   = DIST_TO_REWARD[peak_states]
            ax.hist(distances, bins=20, color=colours[label], edgecolor="white", alpha=0.85)
            ax.axvline(DIST_TO_REWARD[np.argmin(DIST_TO_REWARD)], color="red",
                       linestyle="--", label="reward state")
            ax.set_xlabel("Distance from reward (grid units)")
            ax.set_ylabel("Cell count")
            ax.set_title(f"Episode {ep}")
            ax.legend(fontsize=7)
        fig.suptitle(f"Peak firing distance distribution — {label}", fontsize=11)
        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f"peak_distance_hist_{label}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved: {fname}")


# ── Analysis 4: Mean grid score over training ─────────────────────────────────

def _load_g_rates(ep_path):
    """Load g_rates.npy -> (N_STATES, total_g_cells) or None."""
    fpath = os.path.join(ep_path, "g_rates.npy")
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)


def plot_grid_scores():
    """Mean grid score across all grid cells for both conditions over training.

    For each checkpoint and each condition, every cell's rate map is reshaped
    to (ROOM_W, ROOM_D) and scored with GridScorer.  The mean score across
    all cells is plotted as a time series.  A rising trend in the reward
    condition (or difference from baseline) indicates reward-modulated grid
    cell refinement.
    """
    scorer = GridScorer(ROOM_W)
    colours = {"baseline": "steelblue", "reward_modulated": "darkorange"}

    fig, ax = plt.subplots(figsize=(8, 4))

    for label, plots_dir in [("baseline", BASELINE_DIR), ("reward_modulated", REWARD_DIR)]:
        eps = _episode_dirs(plots_dir)
        ep_list, mean_scores = [], []

        for ep, ep_path in eps:
            g = _load_g_rates(ep_path)
            if g is None:
                continue

            n_cells = g.shape[1]
            scores = []
            for c in range(n_cells):
                cell_map = g[:, c].reshape(ROOM_D, ROOM_W).astype(float)
                if cell_map.max() < 1e-8:
                    continue
                try:
                    _, props = scorer.get_scores(cell_map)
                    scores.append(props["gridscore"])
                except Exception:
                    pass

            if scores:
                ep_list.append(ep)
                mean_scores.append(float(np.mean(scores)))

        if ep_list:
            ax.plot(ep_list, mean_scores, "o-", color=colours[label],
                    label=label, linewidth=2)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean grid score")
    ax.set_title("Mean grid cell score over training (env 0)")
    ax.legend()
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "grid_scores.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# ── Analysis 5: Reward-proximal cell count over loop phase ────────────────────

def plot_proximal_cell_count():
    """Count of place cells whose peak firing is within PROXIMAL_THRESHOLD grid
    units of the reward, restricted to loop-phase checkpoints only
    (episode > LOOP_START_EPISODE).  Compares baseline vs reward_modulated.

    A growing count in the reward condition indicates cells progressively
    reorganising their fields toward the reward site during structured traversal.
    """
    colours = {"baseline": "steelblue", "reward_modulated": "darkorange"}

    fig, ax = plt.subplots(figsize=(8, 4))

    for label, plots_dir in [("baseline", BASELINE_DIR), ("reward_modulated", REWARD_DIR)]:
        eps = _episode_dirs(plots_dir)
        ep_list, counts = [], []

        for ep, ep_path in eps:
            if ep <= LOOP_START_EPISODE:
                continue                         # skip random-phase checkpoints
            p = _load_p_rates(ep_path)
            if p is None:
                continue
            peak_states = np.argmax(p, axis=0)   # (n_cells,)
            distances   = DIST_TO_REWARD[peak_states]
            counts.append(int(np.sum(distances <= PROXIMAL_THRESHOLD)))
            ep_list.append(ep)

        if ep_list:
            ax.plot(ep_list, counts, "o-", color=colours[label],
                    label=label, linewidth=2, markersize=6)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Cells with peak ≤ {PROXIMAL_THRESHOLD} units from reward")
    ax.set_title(
        f"Reward-proximal place cells during loop phase\n"
        f"(episodes > {LOOP_START_EPISODE}, threshold = {PROXIMAL_THRESHOLD} grid units)"
    )
    ax.legend()
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "proximal_cell_count.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# ── Analysis 6: Reward-zone place-field enrichment (pooled across all envs) ───
#
# Unlike the peak-distance metrics above (single argmax peak per cell, no
# null model, env 0 only), this maps onto Hollup et al. 2001's actual result:
# a density-based enrichment ratio (fraction of *fields*, not cells, near
# the goal vs. the fraction of arena area the "near goal" zone occupies)
# tested against a shuffle null, pooled across every environment.
#
# Data comes from tem_probe_eval_multienv.py's single long frozen-weight
# probe walk (results_sim/predictive_analysis/probe_<condition>_rates.npz +
# probe_<condition>_meta.pkl) - ONE summary rate-map per environment per
# condition, not a series of training-time checkpoints. That probe is a
# random walk against the *final* trained model and has no relationship to
# any specific training episode, so its data intentionally lives outside
# results_sim/<condition>/plots/ (which _episode_dirs() above scans for the
# training-progression analyses) - see that script's docstring.

def _state_xy(room_w, room_d, state_density=1):
    """Grid-cell centre xy for a room of given size.

    Must match DiscreteObjectEnvironment / Whittington2020._compute_reward_state_ids's
    own coordinate grid construction (meshgrid(x_array, y_array), flattened
    row-major) exactly, or state indices won't line up with p_rates rows.
    """
    res_w = int(state_density * room_w)
    res_d = int(state_density * room_d)
    x = np.linspace(-room_w / 2 + 0.5 / state_density, room_w / 2 - 0.5 / state_density, res_w)
    y = np.linspace(-room_d / 2 + 0.5 / state_density, room_d / 2 - 0.5 / state_density, res_d)
    xy = np.stack(np.meshgrid(x, y), axis=-1)  # (res_d, res_w, 2)
    return xy.reshape(-1, 2)


def _load_probe_data(condition):
    """Load one condition's probe_<condition>_rates.npz + _meta.pkl from
    results_sim/predictive_analysis/probe/ (written by
    tem_probe_eval_multienv.py). Returns (rates, meta) or (None, None) if the
    probe hasn't been run yet.
    """
    rates_path = os.path.join(PROBE_DIR, f"probe_{condition}_rates.npz")
    meta_path = os.path.join(PROBE_DIR, f"probe_{condition}_meta.pkl")
    if not os.path.exists(rates_path) or not os.path.exists(meta_path):
        return None, None
    rates = dict(np.load(rates_path))
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)
    return rates, meta


def detect_fields(rate_map_2d, thresh_frac=FIELD_THRESH_FRAC, min_size=MIN_FIELD_SIZE):
    """Connected-component place-field detection on one cell's 2D rate map.

    A field is a contiguous (4-connected) region at or above thresh_frac of
    that cell's own peak rate, at least min_size states. A cell can have 0,
    1, or multiple fields - unlike the argmax-only peak used elsewhere in
    this file.

    Returns list of flattened (row-major) peak-state indices, one per field.
    """
    peak = rate_map_2d.max()
    if peak <= 0:
        return []
    mask = rate_map_2d >= thresh_frac * peak
    labeled, n_labels = ndimage.label(mask)  # default structure = 4-connectivity
    flat_rates = rate_map_2d.reshape(-1)
    flat_labels = labeled.reshape(-1)
    fields = []
    for label_id in range(1, n_labels + 1):
        region = np.where(flat_labels == label_id)[0]
        if region.size < min_size:
            continue
        fields.append(int(region[np.argmax(flat_rates[region])]))
    return fields


def is_place_like(rate_map_1d, selectivity_thresh=SELECTIVITY_THRESH, min_amplitude=MIN_AMPLITUDE):
    """Selectivity filter: exclude diffuse/near-flat cells before field
    counting, so noise units don't dilute the enrichment ratio.
    """
    amplitude = rate_map_1d.max() - rate_map_1d.min()
    if amplitude < min_amplitude:
        return False
    mean = rate_map_1d.mean()
    if mean <= 1e-8:
        return True  # all activity concentrated at zero baseline elsewhere: treat as selective
    return (rate_map_1d.max() / mean) >= selectivity_thresh


def _env_field_distances(p_rates_env, room_w, room_d, state_density, reward_location,
                          thresh_frac=FIELD_THRESH_FRAC, min_size=MIN_FIELD_SIZE,
                          selectivity_thresh=SELECTIVITY_THRESH):
    """Detect place-like cells' field distances-from-reward for one environment.

    Returns
    -------
    field_dists : list of float, distance-from-reward (grid units) of every
        detected field's peak state, pooled across this env's place-like cells.
    dist_to_reward : (n_states,) distance-from-reward for every state in this
        env's own geometry - used by the shuffle null.
    """
    xy = _state_xy(room_w, room_d, state_density)
    dist_to_reward = np.linalg.norm(xy - np.array(reward_location), axis=1)

    field_dists = []
    for c in range(p_rates_env.shape[1]):
        col = p_rates_env[:, c]
        if not is_place_like(col, selectivity_thresh):
            continue
        cell_2d = col.reshape(room_d, room_w)
        for peak_state in detect_fields(cell_2d, thresh_frac, min_size):
            field_dists.append(float(dist_to_reward[peak_state]))
    return field_dists, dist_to_reward


def _condition_probe_stats(condition, zone_radius=ZONE_RADIUS,
                            thresh_frac=FIELD_THRESH_FRAC, min_size=MIN_FIELD_SIZE,
                            selectivity_thresh=SELECTIVITY_THRESH):
    """Pool field distances-from-reward across every env, from one condition's
    single probe_<condition>_rates.npz (see _load_probe_data). Returns None if
    the probe hasn't been run for this condition yet.
    """
    rates, meta = _load_probe_data(condition)
    if rates is None:
        return None
    n_envs = len(rates)
    all_field_dists = []
    env_dist_arrays = []
    n_fields_per_env = []
    for j in range(n_envs):
        p_env = rates[f"env{j}"]
        field_dists, dist_to_reward = _env_field_distances(
            p_env,
            meta["room_widths"][j], meta["room_depths"][j], meta["state_densities"][j],
            meta["reward_location"], thresh_frac, min_size, selectivity_thresh,
        )
        all_field_dists.extend(field_dists)
        env_dist_arrays.append(dist_to_reward)
        n_fields_per_env.append(len(field_dists))
    zone_area = sum(int(np.sum(d <= zone_radius)) for d in env_dist_arrays)
    total_area = sum(d.shape[0] for d in env_dist_arrays)
    return {
        "field_dists": all_field_dists,
        "env_dist_arrays": env_dist_arrays,
        "n_fields_per_env": n_fields_per_env,
        "zone_area": zone_area,
        "total_area": total_area,
    }


def _enrichment_ratio(field_dists, zone_area, total_area, zone_radius=ZONE_RADIUS):
    """Hollup-style density enrichment ratio:
        (fields in zone / total fields) / (zone area / total area)
    ratio ~ 1 -> fields uniformly spread; > 1 -> over-represented near reward.
    """
    n_total = len(field_dists)
    if n_total == 0 or zone_area == 0 or total_area == 0:
        return None
    expected_frac = zone_area / total_area
    if expected_frac == 0:
        return None
    n_in_zone = sum(1 for d in field_dists if d <= zone_radius)
    return (n_in_zone / n_total) / expected_frac, n_in_zone, n_total


def _shuffle_null(env_dist_arrays, n_fields_per_env, zone_radius=ZONE_RADIUS,
                   n_shuffles=N_SHUFFLES, rng=None):
    """Null distribution of the pooled enrichment ratio.

    Resamples each env's *observed* field count uniformly among that env's
    own valid states (stratified per env, since room sizes/areas differ) -
    "same amount of data, random locations". Compares the real ratio against
    this to ask whether the observed concentration near reward exceeds chance.
    """
    if rng is None:
        rng = np.random.default_rng(SHUFFLE_SEED)
    zone_area = sum(int(np.sum(d <= zone_radius)) for d in env_dist_arrays)
    total_area = sum(d.shape[0] for d in env_dist_arrays)
    n_total = sum(n_fields_per_env)
    if n_total == 0 or total_area == 0:
        return np.array([])
    expected_frac = zone_area / total_area
    if expected_frac == 0:
        return np.array([])

    null_ratios = np.empty(n_shuffles)
    for s in range(n_shuffles):
        n_in_zone = 0
        for dist_arr, n_fields in zip(env_dist_arrays, n_fields_per_env):
            if n_fields == 0:
                continue
            sampled = rng.integers(0, dist_arr.shape[0], size=n_fields)
            n_in_zone += int(np.sum(dist_arr[sampled] <= zone_radius))
        null_ratios[s] = (n_in_zone / n_total) / expected_frac
    return null_ratios


def plot_reward_zone_enrichment(zone_radius=ZONE_RADIUS, n_shuffles=N_SHUFFLES,
                                 thresh_frac=FIELD_THRESH_FRAC, min_size=MIN_FIELD_SIZE,
                                 selectivity_thresh=SELECTIVITY_THRESH):
    """Reward-zone place-field enrichment ratio, pooled across all envs, one
    number per condition - with a shuffle-based null. Requires
    tem_probe_eval_multienv.py to have been run against both conditions'
    trained agents first (writes results_sim/predictive_analysis/
    probe_<condition>_rates.npz + _meta.pkl).
    """
    rng = np.random.default_rng(SHUFFLE_SEED)
    colours = {"baseline": "steelblue", "reward_modulated": "darkorange"}

    summary = {}
    for label in ("baseline", "reward_modulated"):
        s = _condition_probe_stats(label, zone_radius, thresh_frac, min_size, selectivity_thresh)
        if s is None:
            print(f"No probe data found for {label} — run tem_probe_eval_multienv.py first.")
            continue
        result = _enrichment_ratio(s["field_dists"], s["zone_area"], s["total_area"], zone_radius)
        if result is None:
            print(f"[{label}] not enough fields/area for an enrichment ratio — skipping.")
            continue
        ratio, n_in_zone, n_total = result
        null = _shuffle_null(s["env_dist_arrays"], s["n_fields_per_env"],
                              zone_radius, n_shuffles, rng)
        p_value = float(np.mean(null >= ratio)) if null.size else float("nan")
        summary[label] = {
            "ratio": ratio, "n_in_zone": n_in_zone, "n_total": n_total,
            "null": null, "p_value": p_value, "field_dists": s["field_dists"],
        }
        lo, hi = (np.percentile(null, [5, 95]) if null.size else (float("nan"), float("nan")))
        print(f"[{label}] enrichment ratio = {ratio:.2f} "
              f"({n_in_zone}/{n_total} fields in zone), "
              f"null 5-95pct = [{lo:.2f}, {hi:.2f}], p = {p_value:.4f}")

    if not summary:
        return summary

    # ── Bar plot: observed ratio vs shuffle null, one bar per condition ───────
    keys = [k for k in ("baseline", "reward_modulated") if k in summary]
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, label in enumerate(keys):
        s = summary[label]
        ax.bar(i, s["ratio"], color=colours.get(label, "gray"), alpha=0.85, width=0.6)
        if s["null"].size:
            lo, hi = np.percentile(s["null"], [5, 95])
            ax.plot([i, i], [lo, hi], color="black", linewidth=1.5)
            ax.plot([i - 0.15, i + 0.15], [lo, lo], color="black", linewidth=1.5)
            ax.plot([i - 0.15, i + 0.15], [hi, hi], color="black", linewidth=1.5)
        ax.text(i, s["ratio"] + 0.05, f"p={s['p_value']:.3f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="chance (ratio = 1)")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=9)
    ax.set_ylabel(f"Field density enrichment ratio\n(zone radius = {zone_radius} units)")
    ax.set_title("Reward-zone place-field enrichment (pooled across all envs)\n"
                 "black whiskers = shuffle null 5th-95th percentile")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "reward_zone_enrichment.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")

    # ── Pooled field-distance histogram vs zone radius ────────────────────────
    fig, axs = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), sharey=False)
    if len(keys) == 1:
        axs = [axs]
    for ax, label in zip(axs, keys):
        s = summary[label]
        ax.hist(s["field_dists"], bins=20, color=colours.get(label, "gray"),
                edgecolor="white", alpha=0.85)
        ax.axvline(zone_radius, color="red", linestyle="--", label=f"zone radius ({zone_radius})")
        ax.set_xlabel("Field distance from reward (grid units)")
        ax.set_ylabel("Field count")
        ax.set_title(f"{label}\nratio={s['ratio']:.2f}, p={s['p_value']:.3f}")
        ax.legend(fontsize=7)
    fig.suptitle("Pooled place-field distance from reward (all envs)", fontsize=11)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "reward_zone_field_distance_hist.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")

    return summary


# ── Run all analyses ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running predictive-coding analysis...")
    plot_population_activity_maps()
    plot_value_correlation()
    plot_peak_distance()
    plot_grid_scores()
    plot_proximal_cell_count()
    plot_reward_zone_enrichment()
    print(f"\nAll plots saved to: {OUT_DIR}")
