"""Generate the loop-experiment quantitative report as a plain-text file
under Thesis/quantitative/loop/, pulled directly from the already-computed
CSVs.
"""
import os

import pandas as pd

LOOP_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop"
OUT_PATH = os.path.join(LOOP_DIR, "results.txt")

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


def fmt_table(df, float_cols):
    cols = list(df.columns)
    widths = {c: max(len(c), df[c].astype(str).str.len().max()) for c in cols}
    lines = ["  ".join(c.ljust(widths[c]) for c in cols)]
    lines.append("-" * len(lines[0]))
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append((f"{v:.3f}" if c in float_cols and pd.notna(v) else str(v)).ljust(widths[c]))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def main():
    acc = pd.read_csv(os.path.join(LOOP_DIR, "accuracy_summary.csv"))
    acc.insert(1, "description", acc["condition"].map(describe))
    value_decoding = pd.read_csv(os.path.join(LOOP_DIR, "value_decoding_pre_post.csv"))

    parts = []
    parts.append("LOOP-SHIFT QUANTITATIVE RESULTS")
    parts.append(
        "TEM (Whittington 2020) place-cell probe: all 16 envs forced to uniform 10x10 rooms, agent scripted "
        "around a 20-state ring (border of the inner 6x6 region - the innermost ring that still passes through "
        "the TRUE reward state, since reward_location=[3,3] resolves to an interior point for every room size "
        "this project uses). 5 value landmarks sit at FIXED, spaced-out distances before reward (1,3,5,7,9 "
        "ring-states away, not contiguous); the ring-state right after reward is always kept open. The "
        "remaining ring states are a pool where 5 hold controls (randomly drawn) and the rest are filler; the "
        "rest of each 10x10 room (80 states) is filler too. 500 episodes/seed, 5 seeds (42,7,1,2,3) pooled. At "
        "episode 250, the whole landmark+reward block shifts FORWARD by one ring-state into the open slot. For "
        "the 4 rotation-ON conditions (baseline, rm, baseline_lc, rm_lc), which pool states hold controls is "
        "re-randomized every 25 episodes (matching how those checkpoints were actually trained); for the 4 "
        "\"noshuffle\" conditions that draw is fixed for the whole run. See loop_layout.png for a diagram. "
        "'pre' = episodes 0-249 (original placement), 'post' = episodes 250-499 (after the shift)."
    )
    parts.append("")
    parts.append("=" * 100)
    parts.append("TABLE 1: Object-prediction accuracy (x_gt) by condition, category, and pre/post shift")
    parts.append("=" * 100)
    parts.append("")
    parts.append(fmt_table(acc, float_cols={"acc_landmark", "acc_control", "acc_reward", "acc_filler"}))
    parts.append("")
    parts.append("Note: acc_reward is the hardest category throughout (fewest visits per lap - the loop only ")
    parts.append("passes through the single reward state once per circuit); acc_filler reflects the pool states ")
    parts.append("not currently holding a control, plus everything off the ring.")
    parts.append("")
    parts.append("=" * 100)
    parts.append("TABLE 2: Pooled confusion matrices (true category x predicted category, 5 seeds summed)")
    parts.append("=" * 100)
    parts.append("See <condition>_confusion_{pre,post}_pooled.csv (raw counts) and the _rowpct.csv variant ")
    parts.append("(row-normalized, i.e. P(predicted | true)) for each of the 8 conditions, saved alongside this file.")
    parts.append("")
    parts.append("=" * 100)
    parts.append("TABLE 3: has_value decoding, pre vs post shift - INVALID, see caveat below")
    parts.append("=" * 100)
    parts.append(
        "CAVEAT: every single test below returned a perfect 1.000 (accuracy, balanced accuracy, AUC, and "
        "rm_real/rm_control/bl_real/bl_control rates all exactly 1/1/0/0) - STILL true even after randomizing "
        "which pool states hold controls (and reshuffling that draw every 25 episodes for the rotation-ON "
        "conditions). This confirms the issue was never about controls being fixed - it's that LANDMARKS are "
        "still at the exact same 5 ring positions in every env, every seed, the whole run (required for the "
        "shift mechanic to have consistent, interpretable spacing). Since place cells encode raw position "
        "almost perfectly (established earlier in this project - Finding 1), a classifier asked to separate "
        "landmark-position rows from control-position rows can do so with pure position information alone - it "
        "never needs to touch value, since 'is this one of the 5 fixed landmark slots' is a perfectly separable "
        "question regardless of what the controls are doing. Fixing this would require randomizing landmark "
        "positions too, which conflicts with the fixed approach-spacing the shift experiment needs. Table kept "
        "for the record, not as evidence of (or against) value decodability."
    )
    parts.append("")
    parts.append(fmt_table(
        value_decoding[["test", "half", "accuracy", "balanced_accuracy", "auc", "permutation_p_value",
                         "rate_rm_real", "rate_rm_control", "rate_bl_real", "rate_bl_control"]],
        float_cols={"accuracy", "balanced_accuracy", "auc", "permutation_p_value",
                    "rate_rm_real", "rate_rm_control", "rate_bl_real", "rate_bl_control"},
    ))
    parts.append("")
    parts.append("See non_lc_loop_cycle.png / lc_loop_cycle.png / non_lc_noshuffle_loop_cycle.png / "
                  "lc_noshuffle_loop_cycle.png for the cycle plots (accuracy by ring position, pre vs post).")

    text = "\n".join(parts)
    with open(OUT_PATH, "w") as f:
        f.write(text)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
