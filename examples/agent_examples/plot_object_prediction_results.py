"""Plotting for tem_object_prediction_probe.py's output CSVs.

Reads the --out-raw / --out-layout / --out-moveinfo CSVs (one set per seed)
and produces, per seed, under <out_dir>/seed_<N>/:

  confusion_matrices.png       row-normalised P(predicted | actual), one
                                panel per (window, condition) - a 2x2 grid.
  cumulative_accuracy.png      accuracy at the moved landmark's NEW location,
                                by visit number since the move (does the
                                Hebbian memory re-learn the association with
                                repeated exposure, even with frozen weights?).
  rolling_accuracy_prewindow.png  accuracy in a rolling window across the
                                PRE-move period only - checks for the cold-start
                                transient (Hebbian memory is built online during
                                the probe walk, not carried over from training,
                                so early-window accuracy is expected to climb
                                from ~0 to a plateau rather than being flat).
  after_landmark_visit.png     accuracy 1 and 2 steps after any visit to the
                                moved landmark's (pre- or post-move) location,
                                both conditions, both windows - pools every
                                visit rather than fixating on one state id.
  move_map.png                 env-0 grid coloured by category, old (circle)
                                and new (square) position of the moved
                                landmark connected by an arrow.
  timeline_at_landmark.png,
  timeline_next_state.png      dot-strip: N visits before / N after the move,
                                green = correct, red = incorrect, incorrect
                                dots labelled "P<predicted> A<actual>".

Usage
-----
    cd examples/agent_examples
    python plot_object_prediction_results.py --seeds 42 123 \\
        --data-dir ../../experiments/random_checkpoints/datasets \\
        --out-dir ../../experiments/random_checkpoints/object_pred_plots
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical palette: consistent across every plot in this module.
COLOR_BASELINE = "#4C72B0"
COLOR_REWARD = "#DD8452"
COLOR_CORRECT = "#2CA02C"
COLOR_INCORRECT = "#D62728"
CATEGORY_COLORS = {
    "decoy": "#D9D9D9",
    "control": "#8C8C8C",
    "value_nonreward": "#4C72B0",
    "value_reward": "#DD8452",
}
N_TIMELINE_VISITS = 10


def _load(data_dir, seed):
    raw = pd.read_csv(os.path.join(data_dir, f"object_pred_seed{seed}_raw.csv"))
    layout = pd.read_csv(os.path.join(data_dir, f"object_pred_seed{seed}_layout.csv"))
    moveinfo = pd.read_csv(os.path.join(data_dir, f"object_pred_seed{seed}_moveinfo.csv"))
    return raw, layout, moveinfo


def plot_confusion_matrices(raw, out_path, seed):
    conditions = sorted(raw["condition"].unique())
    windows = ["pre", "post"]
    object_ids = sorted(set(raw["predicted_object_id"]) | set(raw["actual_object_id"]))

    fig, axes = plt.subplots(len(windows), len(conditions), figsize=(6 * len(conditions), 6 * len(windows)))
    axes = np.atleast_2d(axes)
    for i, window in enumerate(windows):
        for j, condition in enumerate(conditions):
            ax = axes[i, j]
            sub = raw[(raw["window"] == window) & (raw["condition"] == condition)]
            mat = pd.crosstab(sub["actual_object_id"], sub["predicted_object_id"])
            mat = mat.reindex(index=object_ids, columns=object_ids, fill_value=0)
            row_sums = mat.sum(axis=1).replace(0, 1)
            mat_norm = mat.div(row_sums, axis=0)
            im = ax.imshow(mat_norm.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_title(f"{condition} — {window}-move", fontsize=10)
            ax.set_xlabel("predicted object id")
            ax.set_ylabel("actual object id")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(predicted | actual)")
    fig.suptitle(f"Seed {seed}: object-prediction confusion matrices", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cumulative_accuracy(raw, moveinfo, out_path, seed):
    """Accuracy at the moved landmark's NEW location, by visit number since
    the move — does frozen-weight Hebbian memory re-learn the association
    with repeated exposure alone?
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for condition, color in [("baseline", COLOR_BASELINE), ("reward_modulated", COLOR_REWARD)]:
        move_row = moveinfo[moveinfo["condition"] == condition].iloc[0]
        to_state = move_row["to_state_id"]
        move_episode = move_row["move_episode"]
        sub = raw[
            (raw["condition"] == condition)
            & (raw["state_id"] == to_state)
            & (raw["episode"] >= move_episode)
        ].sort_values("step")
        if sub.empty:
            continue
        visit_num = np.arange(1, len(sub) + 1)
        cum_acc = np.cumsum(sub["correct"].values) / visit_num
        ax.plot(visit_num, cum_acc, label=condition, color=color, marker="o", markersize=3)
    ax.set_xlabel("visit number to the landmark's new location (since move)")
    ax.set_ylabel("cumulative accuracy")
    ax.set_title(f"Seed {seed}: re-learning the moved landmark's identity")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rolling_accuracy_prewindow(raw, out_path, seed, window_size=500):
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition, color in [("baseline", COLOR_BASELINE), ("reward_modulated", COLOR_REWARD)]:
        sub = raw[(raw["condition"] == condition) & (raw["window"] == "pre")].sort_values("step")
        if len(sub) < window_size:
            continue
        rolling = sub["correct"].rolling(window_size, min_periods=1).mean()
        ax.plot(sub["step"].values, rolling.values, label=condition, color=color)
    ax.set_xlabel("step (pre-move window only)")
    ax.set_ylabel(f"rolling accuracy (window={window_size})")
    ax.set_title(
        f"Seed {seed}: cold-start transient check\n"
        "(Hebbian memory is rebuilt online during this probe walk, not carried over from training)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_after_landmark_visit(raw, moveinfo, out_path, seed):
    """Accuracy 1 and 2 steps after any visit to the moved landmark's
    current-at-the-time location, pooled across every visit (not fixed to
    one state id) — a cleaner baseline-vs-reward_modulated comparison than
    the single-state cumulative-accuracy plot above.
    """
    results = []
    for condition in ("baseline", "reward_modulated"):
        move_row = moveinfo[moveinfo["condition"] == condition].iloc[0]
        from_state, to_state, move_episode = (
            move_row["from_state_id"], move_row["to_state_id"], move_row["move_episode"],
        )
        for window_name, mask_fn in [
            ("pre", lambda df: df["episode"] < move_episode),
            ("post", lambda df: df["episode"] >= move_episode),
        ]:
            sub = raw[(raw["condition"] == condition)].sort_values("step").reset_index(drop=True)
            landmark_state = from_state if window_name == "pre" else to_state
            visit_idx = sub.index[mask_fn(sub) & (sub["state_id"] == landmark_state)]
            for offset in (1, 2):
                idxs = visit_idx + offset
                idxs = idxs[idxs < len(sub)]
                if len(idxs) == 0:
                    continue
                acc = sub.loc[idxs, "correct"].mean()
                results.append({
                    "condition": condition, "window": window_name,
                    "steps_after": offset, "accuracy": acc, "n": len(idxs),
                })
    df = pd.DataFrame(results)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = [f"{r.window}\n+{r.steps_after} step" for r in df.itertuples()]
    x = np.arange(len(df))
    colors = [COLOR_BASELINE if c == "baseline" else COLOR_REWARD for c in df["condition"]]
    ax.bar(x, df["accuracy"], color=colors)
    for xi, n in zip(x, df["n"]):
        ax.text(xi, 0.02, f"n={n}", ha="center", fontsize=7, rotation=90, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Seed {seed}: prediction accuracy after visiting the moved landmark")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BASELINE, label="baseline"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_REWARD, label="reward_modulated"),
    ]
    ax.legend(handles=handles)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_move_map(layout, moveinfo, out_path, seed):
    condition = "baseline"
    sub = layout[(layout["condition"] == condition) & (layout["phase"] == "pre")]
    move_row = moveinfo[moveinfo["condition"] == condition].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    for category, color in CATEGORY_COLORS.items():
        cat_sub = sub[sub["category"] == category]
        ax.scatter(cat_sub["x"], cat_sub["y"], c=color, s=120, marker="s", label=category, edgecolors="white")
    ax.scatter([move_row["from_x"]], [move_row["from_y"]], facecolors="none",
               edgecolors="black", s=300, linewidths=2, marker="o", label="old position")
    ax.scatter([move_row["to_x"]], [move_row["to_y"]], facecolors="none",
               edgecolors="black", s=300, linewidths=2, marker="s", label="new position")
    ax.annotate(
        "", xy=(move_row["to_x"], move_row["to_y"]), xytext=(move_row["from_x"], move_row["from_y"]),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
    )
    ax.set_title(f"Seed {seed}: landmark {move_row['moved_object_id']} relocation (env 0)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _timeline_plot(sub_before, sub_after, out_path, title, label_col_pred, label_col_actual):
    n_before = min(N_TIMELINE_VISITS, len(sub_before))
    n_after = min(N_TIMELINE_VISITS, len(sub_after))
    before = sub_before.tail(n_before)
    after = sub_after.head(n_after)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    x = 0
    for _, row in before.iterrows():
        color = COLOR_CORRECT if row["correct"] else COLOR_INCORRECT
        ax.scatter(x, 0, c=color, s=200, zorder=3)
        if not row["correct"]:
            ax.text(x, 0.15, f"P{row[label_col_pred]}\nA{row[label_col_actual]}",
                    ha="center", fontsize=6, rotation=0)
        x += 1
    ax.axvline(x - 0.5, color="black", linestyle="--", linewidth=1)
    for _, row in after.iterrows():
        color = COLOR_CORRECT if row["correct"] else COLOR_INCORRECT
        ax.scatter(x, 0, c=color, s=200, zorder=3)
        if not row["correct"]:
            ax.text(x, 0.15, f"P{row[label_col_pred]}\nA{row[label_col_actual]}",
                    ha="center", fontsize=6, rotation=0)
        x += 1
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-1, x)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timelines(raw, moveinfo, out_dir, seed):
    for condition in ("baseline", "reward_modulated"):
        move_row = moveinfo[moveinfo["condition"] == condition].iloc[0]
        to_state, move_episode = move_row["to_state_id"], move_row["move_episode"]
        sub = raw[raw["condition"] == condition].sort_values("step").reset_index(drop=True)

        # At-landmark: visits to the (pre/post) landmark location itself -
        # "actual" is always the moved object's id by construction.
        at_landmark = sub[sub["state_id"] == to_state]
        before = at_landmark[at_landmark["episode"] < move_episode]
        after = at_landmark[at_landmark["episode"] >= move_episode]
        _timeline_plot(
            before, after,
            os.path.join(out_dir, f"timeline_at_landmark_{condition}.png"),
            f"Seed {seed} ({condition}): prediction AT the landmark's new location "
            f"({N_TIMELINE_VISITS} visits before/after move)",
            "predicted_object_id", "actual_object_id",
        )

        # Next-state: the step immediately after any at-landmark visit -
        # "actual" varies by visit (agent walks to a different neighbour
        # each time), so label both predicted AND actual on every incorrect
        # dot, not just predicted.
        next_idx = at_landmark.index + 1
        next_idx = next_idx[next_idx < len(sub)]
        next_state_visits = sub.loc[next_idx]
        before_ns = next_state_visits[next_state_visits["episode"] < move_episode]
        after_ns = next_state_visits[next_state_visits["episode"] >= move_episode]
        _timeline_plot(
            before_ns, after_ns,
            os.path.join(out_dir, f"timeline_next_state_{condition}.png"),
            f"Seed {seed} ({condition}): prediction ONE STEP AFTER visiting the landmark "
            f"({N_TIMELINE_VISITS} visits before/after move)",
            "predicted_object_id", "actual_object_id",
        )


def run_seed(data_dir, out_dir, seed):
    print(f"Seed {seed}...")
    raw, layout, moveinfo = _load(data_dir, seed)
    seed_out = os.path.join(out_dir, f"seed_{seed}")
    os.makedirs(seed_out, exist_ok=True)

    plot_confusion_matrices(raw, os.path.join(seed_out, "confusion_matrices.png"), seed)
    plot_cumulative_accuracy(raw, moveinfo, os.path.join(seed_out, "cumulative_accuracy.png"), seed)
    plot_rolling_accuracy_prewindow(raw, os.path.join(seed_out, "rolling_accuracy_prewindow.png"), seed)
    plot_after_landmark_visit(raw, moveinfo, os.path.join(seed_out, "after_landmark_visit.png"), seed)
    plot_move_map(layout, moveinfo, os.path.join(seed_out, "move_map.png"), seed)
    plot_timelines(raw, moveinfo, seed_out, seed)
    print(f"  -> {seed_out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--data-dir", required=True, help="Directory containing object_pred_seed<N>_{raw,layout,moveinfo}.csv")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for seed in args.seeds:
        run_seed(args.data_dir, args.out_dir, seed)
    print("All done.")


if __name__ == "__main__":
    main()
