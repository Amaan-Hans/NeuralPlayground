"""Side-by-side comparison of the already-computed accuracy_summary.csv
from the original campaign (Thesis/quantitative/loop/) and the decoy-tiled
campaign (Thesis2/quantitative/loop/) - no rerun, both already exist.
Prints + saves a clean table: condition x half x category, original vs
decoytile vs delta.
"""
import os

import pandas as pd

ORIG = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop\accuracy_summary.csv"
DECOY = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis2\quantitative\loop\accuracy_summary.csv"
OUT = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis2\quantitative\loop\accuracy_compare_decoy_vs_original.csv"

CATEGORIES = ["landmark", "control", "reward", "filler"]
CONDITIONS = ["baseline", "rm", "baseline_lc", "rm_lc",
              "baseline_noshuffle", "rm_noshuffle", "baseline_lc_noshuffle", "rm_lc_noshuffle"]


def main():
    orig = pd.read_csv(ORIG).set_index(["condition", "half"])
    decoy = pd.read_csv(DECOY).set_index(["condition", "half"])

    rows = []
    for cond in CONDITIONS:
        for half in ("pre", "post"):
            o = orig.loc[(cond, half)]
            d = decoy.loc[(cond, half)]
            row = {"condition": cond, "half": half}
            for c in CATEGORIES:
                row[f"{c}_orig"] = o[f"acc_{c}"]
                row[f"{c}_decoy"] = d[f"acc_{c}"]
                row[f"{c}_delta"] = d[f"acc_{c}"] - o[f"acc_{c}"]
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    # Pretty printed view, one block per category.
    pd.set_option("display.width", 160)
    for c in CATEGORIES:
        print(f"\n=== {c} accuracy: original vs decoy-tiled ===")
        view = out[["condition", "half", f"{c}_orig", f"{c}_decoy", f"{c}_delta"]].copy()
        view.columns = ["condition", "half", "orig", "decoy", "delta"]
        print(view.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
