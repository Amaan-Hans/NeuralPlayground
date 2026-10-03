"""Submit TEM-R training experiments via submitit — to the Wits bigbatch
Slurm cluster for real runs, or locally (no Slurm needed) for smoke tests.

Each (seed, condition) pair becomes one job running whittington_2020_run.py
with matching env vars, so a 2-seed run submits 4 jobs (2 seeds x
{baseline, reward_modulated}) — independently schedulable, matching how
Slurm actually allocates resources (rather than forcing both conditions
onto one node the way run_full_experiment.py's local subprocess-pair does).

bigbatch conventions (see Useful_info/PIPELINE_REBUILD_GUIDE.md): partition
bigbatch, QoS mss_bigbatch, max 6 concurrent jobs/user, 12 nodes/user, 3-day
max walltime, 14 cpus/node. Not every bigbatch node has a working GPU —
training falls back to CPU automatically (much slower) if none is available,
matching Whittington2020's own device-detection.

NEVER run this (or any training) directly via a bare `ssh ... command` on
the login node — always go through submitit's Slurm executor (this script's
default) or srun/sbatch directly. The login node is shared and will just
silently stall.

Usage
-----
    cd examples/agent_examples

    # Local smoke test — no cluster/Slurm involvement, runs in local
    # subprocesses, blocks until done. Do this BEFORE ever submitting to
    # bigbatch, to catch config mistakes without spending queue time.
    python submit_experiments.py --local --test --seeds 42

    # Real cluster submission — run this FROM the cluster (tem_env active,
    # inside examples/agent_examples/). Returns immediately; jobs run
    # asynchronously on bigbatch. Check progress with `squeue -u $USER`.
    python submit_experiments.py --seeds 42 123 --arena-side 10

    # A parameter sweep across arena sizes, still one call:
    python submit_experiments.py --seeds 42 --arena-side 5
    python submit_experiments.py --seeds 42 --arena-side 10 --no-rotate   # control condition
"""

import argparse
import os
import platform
import subprocess
import sys

import submitit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def run_training_job(env_overrides: dict):
    """Runs inside the submitit job's own process (a compute node for a real
    Slurm submission, or a local subprocess for --local). Invokes
    whittington_2020_run.py as a fresh subprocess — identical to how
    run_full_experiment.py already launches it, and keeps job state from
    ever leaking between jobs that might share a worker process.
    """
    env = os.environ.copy()
    env.update(env_overrides)
    result = subprocess.run([sys.executable, "whittington_2020_run.py"], cwd=HERE, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"whittington_2020_run.py failed (env={env_overrides})")
    return env_overrides


def build_env_overrides(seed: int, condition: str, args: argparse.Namespace) -> dict:
    tag = f"_{args.run_tag}" if args.run_tag else ""
    save_root = os.path.join(
        REPO_ROOT, "experiments", "random", f"seed_{seed}{tag}" + ("_test" if args.test else "")
    )
    os.makedirs(save_root, exist_ok=True)
    overrides = {
        "TEM_TEST_MODE": "1" if args.test else "0",
        "TEM_SEED": str(seed),
        "TEM_SAVE_ROOT": save_root,
        "TEM_USE_REWARD": "1" if condition == "reward_modulated" else "0",
        "TEM_VARY_ARENA_SIZE": "0" if args.no_vary_arena_size else "1",
        "TEM_ARENA_SIDE": str(args.arena_side),
        "TEM_ROTATE_ENVIRONMENTS": "0" if args.no_rotate else "1",
        "TEM_N_CONTROL_LANDMARKS": str(args.n_control_landmarks),
        "TEM_DECOY_OBJECT_ID": "" if args.decoy_object_id < 0 else str(args.decoy_object_id),
        "TEM_SAVE_MULTIENV_CSV": "0" if args.no_multienv_csv else "1",
        "TEM_BATCH_SIZE": str(args.batch_size),
    }
    if args.size_cycle is not None:
        overrides["TEM_SIZE_CYCLE"] = args.size_cycle
    if args.load_checkpoint is not None:
        overrides["TEM_LOAD_CHECKPOINT"] = args.load_checkpoint.format(seed=seed, condition=condition)
    if args.n_episode is not None:
        overrides["TEM_N_EPISODE"] = str(args.n_episode)
    if args.eval_interval is not None:
        overrides["TEM_EVAL_INTERVAL"] = str(args.eval_interval)
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="Trajectory seeds to run — one baseline+reward_modulated job pair per seed.",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run locally instead of submitting to Slurm (no cluster needed) — for smoke "
             "tests. Blocks until every job finishes; a real Slurm submission returns immediately.",
    )
    parser.add_argument("--test", action="store_true", help="10-episode smoke test.")
    parser.add_argument(
        "--no-vary-arena-size", action="store_true",
        help="Use one uniform arena size (--arena-side) for all 16 batch slots instead of the "
             "historical 8x8/10x10/12x12-repeating mix (ON by default).",
    )
    parser.add_argument("--arena-side", type=float, default=10.0,
                         help="Only used with --no-vary-arena-size.")
    parser.add_argument(
        "--size-cycle", default=None,
        help="Comma-separated arena side lengths to cycle across the 16 batch slots, e.g. "
             "'5,6,7' or '5,6,7,10,12' — overrides --no-vary-arena-size/--arena-side entirely. "
             "Reward location auto-shrinks to fit the smallest size in the cycle.",
    )
    parser.add_argument(
        "--load-checkpoint", default=None,
        help="Path template to a saved 'agent' checkpoint file to load weights from before "
             "training starts, for curriculum/transfer-learning experiments (e.g. pretrain on "
             "5x5, continue on 10x10). May contain '{seed}' and '{condition}' placeholders, "
             "filled per job, e.g. "
             "'experiments/random/seed_{seed}_curriculum5x5/{condition}/agent'.",
    )
    parser.add_argument(
        "--run-tag", default=None,
        help="Appended to the save root as seed_<seed>_<tag>/ — needed when submitting more "
             "than one config for the same seed (e.g. different --size-cycle variants) so they "
             "don't collide on the same save directory.",
    )
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--n-control-landmarks", type=int, default=10)
    parser.add_argument("--decoy-object-id", type=int, default=30)
    parser.add_argument("--no-multienv-csv", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-episode", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--partition", default="bigbatch")
    parser.add_argument("--qos", default="mss_bigbatch")
    parser.add_argument(
        "--time-hours", type=float, default=6.0,
        help="Slurm walltime per job, hours (default 6 — a 5000-episode/100-state run without "
             "rotation took ~3h; this budgets headroom for rotation + all-envs CSV overhead).",
    )
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument(
        "--gres", default="gpu:1",
        help="Slurm --gres string (default 'gpu:1'). Set to '' for a CPU-only submission — "
             "check `sinfo -o '%%N %%G'` on the cluster for the actual GRES name if this doesn't match.",
    )
    parser.add_argument("--mem-gb", type=int, default=16)
    parser.add_argument(
        "--log-dir", default=None,
        help="submitit log folder (default: <repo>/experiments/submitit_logs).",
    )
    args = parser.parse_args()

    log_dir = args.log_dir or os.path.join(REPO_ROOT, "experiments", "submitit_logs")
    os.makedirs(log_dir, exist_ok=True)

    if args.local:
        # submitit's LocalExecutor (AutoExecutor(cluster="local")) manages
        # child processes with POSIX-only signals (SIGCONT/SIGKILL), which
        # don't exist on Windows' signal module. On Windows, fall back to
        # DebugExecutor instead — same .submit()/.result() API, runs jobs
        # synchronously in-process rather than as separate OS processes, so
        # it's dependency-free on any platform. On Linux/Mac (including the
        # cluster's own login/dev nodes, if ever used for a local test)
        # LocalExecutor works fine and actually isolates each job as its own
        # process, which is closer to what a real Slurm job looks like.
        if platform.system() == "Windows":
            print("Windows detected — using DebugExecutor (in-process) instead of "
                  "LocalExecutor (POSIX-only signal handling).")
            executor = submitit.DebugExecutor(folder=log_dir)
        else:
            executor = submitit.AutoExecutor(folder=log_dir, cluster="local")
            executor.update_parameters(timeout_min=int(args.time_hours * 60))
    else:
        executor = submitit.AutoExecutor(folder=log_dir, cluster="slurm")
        executor.update_parameters(
            slurm_partition=args.partition,
            slurm_qos=args.qos,
            slurm_gres=args.gres if args.gres else None,
            slurm_time=int(args.time_hours * 60),
            slurm_cpus_per_task=args.cpus_per_task,
            slurm_mem=f"{args.mem_gb}G",
            slurm_job_name="tem_r",
            # Without this, submitit's generated sbatch script never sets
            # --ntasks/--ntasks-per-node at all, and this cluster's Slurm
            # defaults then launch the srun command multiple times within
            # the SAME job allocation (observed: 3 concurrent copies of the
            # training script, all racing to os.makedirs() the identical
            # TEM_SAVE_ROOT and crashing 2-of-3 with FileExistsError - the
            # "surviving" copy still isn't a normal single run, it's just
            # whichever one won the race). This is exactly one task, always.
            tasks_per_node=1,
        )

    jobs = []
    for seed in args.seeds:
        for condition in ("baseline", "reward_modulated"):
            overrides = build_env_overrides(seed, condition, args)
            job = executor.submit(run_training_job, overrides)
            jobs.append((seed, condition, job))
            print(f"Submitted seed={seed} condition={condition} -> job id {job.job_id}")

    print(f"\n{len(jobs)} job(s) submitted. Logs: {log_dir}")
    if args.local:
        print("Waiting for local jobs to finish (this blocks; a real Slurm submission returns immediately)...")
        failures = []
        for seed, condition, job in jobs:
            try:
                job.result()
                print(f"  done: seed={seed} condition={condition}")
            except Exception as exc:  # noqa: BLE001 - surface any job failure, then keep checking the rest
                failures.append((seed, condition, exc))
                print(f"  FAILED: seed={seed} condition={condition} -> {exc}")
        if failures:
            raise RuntimeError(f"{len(failures)} job(s) failed: {failures}")
    else:
        print("Jobs queued on Slurm — check progress with `squeue -u $USER`, "
              "or poll job.result() from a script (blocks until that job finishes).")


if __name__ == "__main__":
    main()
