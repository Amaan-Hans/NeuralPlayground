"""Run probe_loop_shift.py's loop-shift probe across all 8 conditions x 5
seeds, saving into Thesis/quantitative/loop/ (episodes CSVs + confusion
matrices per condition/seed). One process, strictly sequential (never run
alongside any other probe - this GPU cannot share across concurrent
processes).
"""
import os

from probe_loop_shift import run_probe

THESIS = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis"
OUT_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop"
SEEDS = [42, 7, 1, 2, 3]

CAMPAIGNS = {
    "baseline": (
        os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj"), "baseline"),
    "rm": (
        os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj"), "reward_modulated"),
    "baseline_lc": (
        os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_lc"), "baseline_lc"),
    "rm_lc": (
        os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_lc"), "reward_modulated_lc"),
    "baseline_noshuffle": (
        os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle"), "baseline"),
    "rm_noshuffle": (
        os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle"), "reward_modulated"),
    "baseline_lc_noshuffle": (
        os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle_lc"), "baseline_lc"),
    "rm_lc_noshuffle": (
        os.path.join(THESIS, "reward_modulated_experiments_ctrlshuffle25_rewardobj_noshuffle_lc"), "reward_modulated_lc"),
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
            run_probe(save_path, label, OUT_DIR)
        print(f"### seed {seed} done ###", flush=True)


if __name__ == "__main__":
    main()
