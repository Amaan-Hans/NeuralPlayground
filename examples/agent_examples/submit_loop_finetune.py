"""Submit whittington_2020_loop_finetune_run.py jobs via submitit, exactly
like submit_experiments.py does for the main random-walk training script.
Each (seed, condition-label) pair becomes one job: loads that seed's already-
trained checkpoint for that condition (from the ORIGINAL ctrlshuffle25_
rewardobj campaign) and continues training it on the ring-loop structure for
TEM_N_EPISODES episodes (default 1000, shift at 500).

Output goes to experiments/random/seed_<seed>_loop_finetune/<cond_label>/ -
a brand-new tag, never collides with any existing experiment directory.
Structured per-seed specifically so more seeds can be added later just by
passing --seeds with more values - only seed 42 exists yet for the base
campaign's checkpoints this reads from, but nothing here hardcodes that.

Usage
-----
    cd examples/agent_examples
    python submit_loop_finetune.py --seeds 42
    python submit_loop_finetune.py --seeds 42 7 --n-episodes 1000 --shift-at-episode 500
"""
import argparse
import os
import subprocess
import sys

import submitit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXPERIMENTS_RANDOM = os.path.join(REPO_ROOT, "experiments", "random")

# cond_label -> (source campaign dir name under experiments/random/seed_<seed>_..., condition subdir)
# Mirrors probe_loop_ring_all.py's CAMPAIGNS keys/structure exactly, pointed at
# the ORIGINAL (non-decoy-tiled) ctrlshuffle25_rewardobj campaign.
CAMPAIGNS = {
    "baseline": ("ctrlshuffle25_rewardobj_50k", "baseline"),
    "rm": ("ctrlshuffle25_rewardobj_50k", "reward_modulated"),
    "baseline_lc": ("ctrlshuffle25_rewardobj_lc_50k", "baseline_lc"),
    "rm_lc": ("ctrlshuffle25_rewardobj_lc_50k", "reward_modulated_lc"),
    "baseline_noshuffle": ("ctrlshuffle25_rewardobj_noshuffle_50k", "baseline"),
    "rm_noshuffle": ("ctrlshuffle25_rewardobj_noshuffle_50k", "reward_modulated"),
    "baseline_lc_noshuffle": ("ctrlshuffle25_rewardobj_noshuffle_lc", "baseline_lc"),
    "rm_lc_noshuffle": ("ctrlshuffle25_rewardobj_noshuffle_lc", "reward_modulated_lc"),
}

# Same 8 cond_labels, pointed at the decoy-tiled campaign's checkpoints instead
# (--base decoytile). Dir names here have no "_50k"/trailing suffix, per the
# actual experiments/random/ layout for this campaign.
CAMPAIGNS_DECOYTILE = {
    "baseline": ("ctrlshuffle25_rewardobj_decoytile", "baseline"),
    "rm": ("ctrlshuffle25_rewardobj_decoytile", "reward_modulated"),
    "baseline_lc": ("ctrlshuffle25_rewardobj_lc_decoytile", "baseline_lc"),
    "rm_lc": ("ctrlshuffle25_rewardobj_lc_decoytile", "reward_modulated_lc"),
    "baseline_noshuffle": ("ctrlshuffle25_rewardobj_noshuffle_decoytile", "baseline"),
    "rm_noshuffle": ("ctrlshuffle25_rewardobj_noshuffle_decoytile", "reward_modulated"),
    "baseline_lc_noshuffle": ("ctrlshuffle25_rewardobj_noshuffle_lc_decoytile", "baseline_lc"),
    "rm_lc_noshuffle": ("ctrlshuffle25_rewardobj_noshuffle_lc_decoytile", "reward_modulated_lc"),
}


def run_finetune_job(env_overrides: dict):
    env = os.environ.copy()
    env.update(env_overrides)
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [sys.executable, "whittington_2020_loop_finetune_run.py"], cwd=HERE, env=env
    )
    if result.returncode != 0:
        raise RuntimeError(f"whittington_2020_loop_finetune_run.py failed (env={env_overrides})")
    return env_overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                         help="Which base-campaign seeds to fine-tune (must already have all 8 "
                              "conditions trained under the ctrlshuffle25_rewardobj campaign).")
    parser.add_argument("--base", default="original", choices=["original", "decoytile"],
                         help="Which trained campaign to fine-tune from: the original "
                              "ctrlshuffle25_rewardobj_50k checkpoints, or the decoy-tiled ones.")
    parser.add_argument("--conditions", nargs="+", default=None,
                         choices=list(set(CAMPAIGNS) | set(CAMPAIGNS_DECOYTILE)))
    parser.add_argument("--n-episodes", type=int, default=1000)
    parser.add_argument("--shift-at-episode", type=int, default=500)
    parser.add_argument("--local", action="store_true", help="Run locally (no Slurm) for a smoke test.")
    parser.add_argument("--partition", default="bigbatch")
    parser.add_argument("--qos", default="mss_bigbatch")
    parser.add_argument("--time-hours", type=float, default=4.0)
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem-gb", type=int, default=16)
    parser.add_argument("--gres", default="")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    campaigns = CAMPAIGNS_DECOYTILE if args.base == "decoytile" else CAMPAIGNS
    conditions = args.conditions or list(campaigns.keys())
    # Separate save-tag for the decoytile base so this can never collide with
    # (or be mistaken for) a fine-tune of the original campaign's checkpoints.
    finetune_tag = "loop_finetune_decoytile" if args.base == "decoytile" else "loop_finetune"

    log_dir = args.log_dir or os.path.join(REPO_ROOT, "experiments", "submitit_logs")
    os.makedirs(log_dir, exist_ok=True)

    if args.local:
        executor = submitit.DebugExecutor(folder=log_dir)
    else:
        executor = submitit.AutoExecutor(folder=log_dir, cluster="slurm")
        executor.update_parameters(
            slurm_partition=args.partition,
            slurm_qos=args.qos,
            slurm_gres=args.gres if args.gres else None,
            slurm_time=int(args.time_hours * 60),
            slurm_cpus_per_task=args.cpus_per_task,
            slurm_mem=f"{args.mem_gb}G",
            slurm_job_name="tem_loopft",
            tasks_per_node=1,
        )

    jobs = []
    for seed in args.seeds:
        for cond_label in conditions:
            campaign_dir, condition = campaigns[cond_label]
            base_checkpoint_dir = os.path.join(EXPERIMENTS_RANDOM, f"seed_{seed}_{campaign_dir}", condition)
            save_root = os.path.join(EXPERIMENTS_RANDOM, f"seed_{seed}_{finetune_tag}", cond_label)
            if os.path.isdir(save_root) and os.listdir(save_root):
                print(f"skip (already has content): {save_root}")
                continue
            rotation_enabled = "noshuffle" not in cond_label
            env_overrides = {
                "TEM_BASE_CHECKPOINT_DIR": base_checkpoint_dir,
                "TEM_SAVE_ROOT": save_root,
                "TEM_SEED": str(seed),
                "TEM_ROTATION_ENABLED": "1" if rotation_enabled else "0",
                "TEM_N_EPISODES": str(args.n_episodes),
                "TEM_SHIFT_AT_EPISODE": str(args.shift_at_episode),
            }
            job = executor.submit(run_finetune_job, env_overrides)
            jobs.append((f"seed={seed} cond={cond_label}", job))
            print(f"Submitted [seed={seed} cond={cond_label}] base={base_checkpoint_dir} -> job id {job.job_id}")

    print(f"\n{len(jobs)} job(s) submitted. Logs: {log_dir}")
    if args.local:
        for label, job in jobs:
            job.result()
            print(f"  done: {label}")
    else:
        print("Jobs queued on Slurm — check progress with `squeue -u $USER`.")


if __name__ == "__main__":
    main()
