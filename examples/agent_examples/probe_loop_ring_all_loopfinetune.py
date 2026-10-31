"""Same as probe_loop_ring_all.py (run probe_loop_ring.py's full 500-episode
ring probe, saving episodes/confusion/position_accuracy/place_cells CSVs),
but pointed at the loop-finetune experiment's checkpoints: seed 42's
original (non-decoy-tiled) ctrlshuffle25_rewardobj_50k checkpoints, each
continued for 1000 more episodes directly on this same ring-loop structure
(real gradient/Hebbian/TD training through a value-block shift at episode
500 - see whittington_2020_loop_finetune_run.py), then downloaded flat into
Thesis2/loop_finetune_seed42/<cond_label>/.

Only 1 seed (42) exists for this experiment, so output filenames use
"_seed42" for consistency with the other two campaigns' naming convention,
not because more seeds are expected. Output goes to
Thesis2/quantitative/loop_finetune/ so it never collides with the original
or decoy-tiled campaigns' probe output.
"""
import os

from probe_loop_ring import run_probe

THESIS2 = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis2"
LOOP_FINETUNE_ROOT = os.path.join(THESIS2, "loop_finetune_seed42")
OUT_DIR = os.path.join(THESIS2, "quantitative", "loop_finetune")
SEED = 42

# cond_label -> rotation_enabled (noshuffle conditions never reshuffle controls)
CONDITIONS = [
    "baseline", "rm", "baseline_lc", "rm_lc",
    "baseline_noshuffle", "rm_noshuffle", "baseline_lc_noshuffle", "rm_lc_noshuffle",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for cond in CONDITIONS:
        label = f"{cond}_seed{SEED}"
        out_episodes = os.path.join(OUT_DIR, f"{label}_episodes.csv")
        if os.path.exists(out_episodes):
            print(f"skip (exists): {out_episodes}")
            continue
        save_path = os.path.join(LOOP_FINETUNE_ROOT, cond)
        rotation_enabled = "noshuffle" not in cond
        run_probe(save_path, label, OUT_DIR, rotation_enabled)
    print("### loop-finetune probe done ###", flush=True)


if __name__ == "__main__":
    main()
