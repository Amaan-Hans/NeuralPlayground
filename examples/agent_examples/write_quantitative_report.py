"""Generate the random-walk quantitative report (object-prediction accuracy
by condition/category, and has_value decoding by condition/group) as a
plain-text file under Thesis/quantitative/random-walk/, pulled directly from
the already-computed CSVs so the numbers can't drift from what was actually
run.
"""
import os

import pandas as pd

AGENT_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ACC_DIR = os.path.join(AGENT_EXAMPLES_DIR, "accuracy_curve_results")
OUT_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\random-walk"
OUT_PATH = os.path.join(OUT_DIR, "results.txt")

SEEDS = [42, 7, 1, 2, 3]

# condition label -> (has value?, LC gating?, control-landmark rotation?)
CONDITIONS = {
    "baseline":              ("no",  "no",  "yes"),
    "rm":                    ("yes", "no",  "yes"),
    "baseline_lc":           ("no",  "yes", "yes"),
    "rm_lc":                 ("yes", "yes", "yes"),
    "baseline_noshuffle":    ("no",  "no",  "no"),
    "rm_noshuffle":          ("yes", "no",  "no"),
    "baseline_lc_noshuffle": ("no",  "yes", "no"),
    "rm_lc_noshuffle":       ("yes", "yes", "no"),
}


def describe(cond):
    has_value, lc, rotation = CONDITIONS[cond]
    return f"value={has_value}, LC gating={lc}, control-landmark rotation={rotation}"


def accuracy_table():
    rows = []
    for cond in CONDITIONS:
        frames = [pd.read_csv(os.path.join(ACC_DIR, f"{cond}_seed{s}_accuracycurve.csv")) for s in SEEDS]
        df = pd.concat(frames, ignore_index=True)
        rows.append({
            "condition": cond,
            "description": describe(cond),
            "landmark_accuracy": df["x_gt_landmark"].mean(),
            "control_accuracy": df["x_gt_control"].mean(),
            "filler_accuracy": df["x_gt_filler"].mean(),
        })
    return pd.DataFrame(rows)


def value_decoding_tables():
    overall = pd.read_csv(os.path.join(AGENT_EXAMPLES_DIR, "has_value_all_8_cells_overall.csv"))
    groups = pd.read_csv(os.path.join(AGENT_EXAMPLES_DIR, "has_value_all_8_cells_groups.csv"))

    # Map each paired test's 4 groups back onto individual condition rows
    # (rm_real/rm_control belong to the reward_modulated side, bl_real/
    # bl_control to the baseline side of that same {rotation, LC} pair).
    test_to_conditions = {
        "rotation ON, no LC": ("baseline", "rm"),
        "rotation ON, LC": ("baseline_lc", "rm_lc"),
        "rotation OFF, no LC": ("baseline_noshuffle", "rm_noshuffle"),
        "rotation OFF, LC": ("baseline_lc_noshuffle", "rm_lc_noshuffle"),
    }
    group_to_cond_role = {
        "bl_real": ("baseline", "landmark"), "bl_control": ("baseline", "control"),
        "rm_real": ("rm", "landmark"), "rm_control": ("rm", "control"),
    }

    rows = []
    for test, (bl_cond, rm_cond) in test_to_conditions.items():
        sub = groups[groups["test"] == test]
        for _, r in sub.iterrows():
            role, group_kind = group_to_cond_role[r["group"]]
            cond = bl_cond if role == "baseline" else rm_cond
            rows.append({
                "condition": cond, "description": describe(cond), "group": group_kind,
                "predicted_has_value_rate": r["predicted_positive_rate"],
            })
    per_condition = pd.DataFrame(rows)
    return overall, per_condition


def fmt_table(df, float_cols, col_widths=None):
    cols = list(df.columns)
    widths = {c: max(len(c), df[c].astype(str).str.len().max()) for c in cols}
    if col_widths:
        widths.update(col_widths)
    lines = []
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in float_cols:
                cells.append(f"{v:.3f}".ljust(widths[c]))
            else:
                cells.append(str(v).ljust(widths[c]))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    acc_df = accuracy_table()
    overall_df, per_cond_df = value_decoding_tables()

    parts = []
    parts.append("RANDOM-WALK QUANTITATIVE RESULTS")
    parts.append("TEM (Whittington 2020) place-cell probes, all 8 conditions "
                  "(reward on/off x LC gating on/off x control-landmark rotation on/off), "
                  "5 seeds (42, 7, 1, 2, 3) pooled per condition unless noted.")
    parts.append("")
    parts.append("=" * 100)
    parts.append("TABLE 1: Object-prediction accuracy (x_gt) by condition and object category")
    parts.append("=" * 100)
    parts.append("Frozen-weight probe, 500 episodes/seed, control-landmark rotation (where enabled) every 25 "
                  "episodes. landmark/control/filler = mean one-step predictive accuracy on states whose TRUE "
                  "current object is a value landmark / control landmark / filler-remainder object respectively.")
    parts.append("")
    parts.append(fmt_table(acc_df, float_cols={"landmark_accuracy", "control_accuracy", "filler_accuracy"}))
    parts.append("")
    parts.append("=" * 100)
    parts.append("TABLE 2a: has_value decoding - per-condition predicted-positive rate")
    parts.append("=" * 100)
    parts.append("LogisticRegression on Theta-scale place-cell activity (reward_prediction project's "
                  "has_value_full_training_test.py recipe): positive class = that pair's reward_modulated real "
                  "landmarks; negative class = that pair's own reward_modulated controls + baseline real "
                  "landmarks + baseline controls, trained together so shape/identity alone cannot separate the "
                  "classes. Held out by physical (seed, group, env_id, state_id) location. 'landmark'/'control' "
                  "below is which group within that condition this rate is for.")
    parts.append("")
    parts.append(fmt_table(per_cond_df, float_cols={"predicted_has_value_rate"}))
    parts.append("")
    parts.append("=" * 100)
    parts.append("TABLE 2b: has_value decoding - overall held-out classifier performance per {rotation, LC} pair")
    parts.append("=" * 100)
    parts.append("Each row is one classifier (trained+evaluated on that pair's pooled 4-group population, "
                  "n_test=640, 300-permutation null test - all 4 pairs significant at p=0.0033).")
    parts.append("")
    parts.append(fmt_table(
        overall_df[["test", "accuracy", "balanced_accuracy", "auc", "permutation_p_value", "n_train", "n_test"]],
        float_cols={"accuracy", "balanced_accuracy", "auc", "permutation_p_value"},
    ))
    parts.append("")

    text = "\n".join(parts)
    with open(OUT_PATH, "w") as f:
        f.write(text)
    print(f"Saved {OUT_PATH}")
    print()
    print(text)


if __name__ == "__main__":
    main()
