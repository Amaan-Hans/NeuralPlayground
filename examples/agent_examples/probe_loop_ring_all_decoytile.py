"""Same as probe_loop_ring_all.py (run probe_loop_ring.py across all 8
conditions x 5 seeds, saving episodes/confusion/position_accuracy/place_cells
CSVs), but pointed at the decoy-tiled campaign's checkpoints (Thesis2 /
*_decoytile, downloaded after the decoy-tile retrain finished) instead of the
original Thesis / *_ctrlshuffle25_rewardobj checkpoints. Output goes to
Thesis2/quantitative/loop/ so it never collides with the original results.
"""
import os

from probe_loop_ring import run_probe

THESIS2 = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis2"
OUT_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis2\quantitative\loop"
SEEDS = [42, 7, 1, 2, 3]

CAMPAIGNS = {
    "baseline": (
        os.path.join(THESIS2, "reward_modulated_experiments_ctrlshuffle25_rewardobj_decoytile"), "baseline"),
    "rm": (
        os.path.join(THESIS2, "reward_modulated_experiments_ctrlshuffle25_rewardobj_decoytile"), "reward_modulated"),
    "baseline_lc": (
        os.path.join(THESIS2, "reward_modulated_experiments_ctrlshuffle25_rewardobj_lc_decoytile"), "baseline_lc"),
    "rm_lc": (
        os.path.join(THESIS2, "reward_modulated_experiments_ctrlshuffle25_rewardobj_lc_decoytile"), "reward_modulated_lc"),
    "baseline_noshuffle": (
        os.path.join(THESIS2, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle_decoytile"), "baseline"),
    "rm_noshuffle": (
        os.path.join(THESIS2, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle_decoytile"), "reward_modulated"),
    "baseline_lc_noshuffle": (
        os.path.join(THESIS2, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle_lc_decoytile"), "baseline_lc"),
    "rm_lc_noshuffle": (
        os.path.join(THESIS2, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle_lc_decoytile"), "reward_modulated_lc"),
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for seed in SEEDS:
        for cond, (root, condition) in CAMPAIGNS.items():
            label = f"{cond}_seed{seed}"
            out_episodes = os.path.join(OUT_DIR, f"{label}_episodes.csv")
            if os.path.exists(out_episodes):
                print(f"skip (exists): {out_episodes}")
                continue
            save_path = os.path.join(root, f"seed_{seed}", condition)
            rotation_enabled = "noshuffle" not in cond
            run_probe(save_path, label, OUT_DIR, rotation_enabled)
        print(f"### seed {seed} done ###", flush=True)


if __name__ == "__main__":
    main()
