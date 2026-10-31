"""Generate the decoy-tiled loop-experiment quantitative report as a plain-
text file under Thesis2/quantitative/loop/, pulled directly from the
already-computed CSVs. Same as write_loop_report.py, minus Table 3 (has_value
decoding) - that table comes from probe_loop_value_decoding.py, a different
(square-loop) probe never run against the decoy-tiled checkpoints, and was
already flagged INVALID for the original campaign for unrelated reasons
(perfectly-decodable raw position), so it isn't reproduced here.
"""
import os

import pandas as pd

LOOP_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis2\quantitative\loop"
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

    parts = []
    parts.append("LOOP-SHIFT QUANTITATIVE RESULTS - DECOY-TILED CAMPAIGN")
    parts.append(
        "TEM (Whittington 2020) place-cell probe, identical ring-loop protocol to the original campaign's "
        "loop report (same 20-state ring, same 5 fixed-distance landmarks, same shift-at-episode-250 mechanic, "
        "500 episodes/seed, 5 seeds (42,7,1,2,3) pooled), but run against checkpoints trained with ALL "
        "remainder states (every ring/room slot not holding a landmark, control, or reward) tiled with a single "
        "fixed decoy object id instead of the varied random fillers the original campaign used - i.e. the "
        "'uninformative environment' condition. 'pre' = episodes 0-249 (original placement), 'post' = episodes "
        "250-499 (after the shift)."
    )
    parts.append("")
    parts.append("=" * 100)
    parts.append("TABLE 1: Object-prediction accuracy (x_gt) by condition, category, and pre/post shift")
    parts.append("=" * 100)
    parts.append("")
    parts.append(fmt_table(acc, float_cols={"acc_landmark", "acc_control", "acc_reward", "acc_filler"}))
    parts.append("")
    parts.append("=" * 100)
    parts.append("TABLE 2: Pooled confusion matrices (true category x predicted category, 5 seeds summed)")
    parts.append("=" * 100)
    parts.append("See <condition>_confusion_{pre,post}_pooled.csv (raw counts) and the _rowpct.csv variant ")
    parts.append("(row-normalized, i.e. P(predicted | true)) for each of the 8 conditions, saved alongside this file.")
    parts.append("")
    parts.append("See non_lc_loop_cycle.png / lc_loop_cycle.png / non_lc_noshuffle_loop_cycle.png / "
                  "lc_noshuffle_loop_cycle.png for the cycle plots (accuracy by ring position, pre vs post).")

    text = "\n".join(parts)
    with open(OUT_PATH, "w") as f:
        f.write(text)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
