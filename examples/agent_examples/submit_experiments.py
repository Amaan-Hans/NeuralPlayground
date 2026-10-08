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


def run_training_job_group(env_overrides_list: list):
    """Like run_training_job, but packs several runs into ONE submitit job
    (hence one slot against the account's MaxJobsPerUser QoS cap), each as
    its own subprocess launched concurrently via subprocess.Popen and
    waited on at the end - NOT via slurm_tasks_per_node>1, which previously
    caused multiple copies of the training script to race on the same
    TEM_SAVE_ROOT (see the tasks_per_node=1 comment in main()). That bug
    can't recur here: each subprocess in this group has its own distinct
    env_overrides (different seed/condition), so each gets a distinct
    TEM_SAVE_ROOT - no collision. Exists because the account's job-level
    concurrency cap (6 running at once, checked via `sacctmgr show qos`)
    is the real bottleneck, not per-node CPU/RAM - fewer, fatter jobs that
    each run several trainings in parallel get more real concurrency than
    the same total run count spread across more, thinner jobs ever could,
    given that cap.
    """
    procs = []
    for env_overrides in env_overrides_list:
        env = os.environ.copy()
        env.update(env_overrides)
        proc = subprocess.Popen([sys.executable, "whittington_2020_run.py"], cwd=HERE, env=env)
        procs.append((proc, env_overrides))

    failures = []
    for proc, env_overrides in procs:
        if proc.wait() != 0:
            failures.append(env_overrides)
    if failures:
        raise RuntimeError(f"{len(failures)} run(s) in this group failed: {failures}")
    return env_overrides_list


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
        "TEM_SCALE_WALK_BY_SIZE": "0" if args.no_walk_scale_by_size else "1",
    }
    if args.size_cycle is not None:
        overrides["TEM_SIZE_CYCLE"] = args.size_cycle
    if args.load_checkpoint is not None:
        overrides["TEM_LOAD_CHECKPOINT"] = args.load_checkpoint.format(seed=seed, condition=condition)
    if args.walk_it_min is not None:
        overrides["TEM_WALK_IT_MIN"] = str(args.walk_it_min)
    if args.walk_it_max is not None:
        overrides["TEM_WALK_IT_MAX"] = str(args.walk_it_max)
    if args.n_episode is not None:
        overrides["TEM_N_EPISODE"] = str(args.n_episode)
    if args.eval_interval is not None:
        overrides["TEM_EVAL_INTERVAL"] = str(args.eval_interval)
    if args.control_shuffle_interval is not None:
        overrides["TEM_CONTROL_SHUFFLE_INTERVAL"] = str(args.control_shuffle_interval)
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
    parser.add_argument(
        "--conditions", nargs="+", default=["baseline", "reward_modulated"],
        choices=["baseline", "reward_modulated"],
        help="Which condition(s) to submit (default: both). E.g. --conditions "
             "reward_modulated to resubmit just one condition without re-running the other.",
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
        "--no-walk-scale-by-size", action="store_true",
        help="Disable scaling rotation walk length by environment size (ON by default). Use "
             "this for a flat walk_it_min/max/window across all slots regardless of size, "
             "matching Whittington et al. 2020's STAR Methods (~2000-5000 raw-step dwell per "
             "environment, not scaled by their worlds' 64-127-state size range).",
    )
    parser.add_argument(
        "--walk-it-min", type=int, default=None,
        help="Override params['walk_it_min'] (n_rollout-step chunks, i.e. raw steps / "
             "n_rollout). E.g. with n_rollout=20, --walk-it-min 100 --walk-it-max 250 gives "
             "the paper's literal ~2000-5000 raw-step dwell per environment.",
    )
    parser.add_argument("--walk-it-max", type=int, default=None)
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
    parser.add_argument(
        "--control-shuffle-interval", type=int, default=None,
        help="Re-randomize control-landmark positions (reward-biased, same mechanism as "
             "initial placement) every N episodes, within the SAME environment instance - "
             "value landmarks, decoy/remainder objects, and all agent-side state (Hebbian "
             "memory, TD table, visited-states) are untouched. Independent of --no-rotate "
             "(a full rotation also resets agent-side state; this doesn't). Off by default.",
    )
    parser.add_argument(
        "--runs-per-node", type=int, default=1,
        help="Pack this many (seed, condition) runs into each submitted job, run concurrently "
             "as sibling subprocesses on one node (see run_training_job_group) - NOT via Slurm "
             "tasks_per_node, which previously caused multiple copies to race on the same "
             "TEM_SAVE_ROOT. Exists because the account's MaxJobsPerUser QoS cap limits "
             "CONCURRENT jobs regardless of per-node resources (check with `sacctmgr show qos "
             "<qos-name>`) - fewer, fatter jobs get more real concurrency under that cap than "
             "the same run count spread across more jobs. --cpus-per-task/--mem-gb are each "
             "requested ONCE PER RUN in the group and multiplied up automatically.",
    )
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
            # --cpus-per-task/--mem-gb are PER RUN; a group running
            # --runs-per-node runs concurrently on one node needs that many
            # times the resources, requested as a single Slurm allocation.
            slurm_cpus_per_task=args.cpus_per_task * args.runs_per_node,
            slurm_mem=f"{args.mem_gb * args.runs_per_node}G",
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

    # Build every (seed, condition) run's overrides first, then chunk into
    # groups of --runs-per-node, each submitted as ONE job.
    runs = [
        (seed, condition, build_env_overrides(seed, condition, args))
        for seed in args.seeds
        for condition in args.conditions
    ]
    chunk_size = max(1, args.runs_per_node)
    chunks = [runs[i:i + chunk_size] for i in range(0, len(runs), chunk_size)]

    jobs = []
    for chunk in chunks:
        overrides_list = [c[2] for c in chunk]
        labels = [f"seed={c[0]} condition={c[1]}" for c in chunk]
        if len(chunk) == 1:
            job = executor.submit(run_training_job, overrides_list[0])
        else:
            job = executor.submit(run_training_job_group, overrides_list)
        jobs.append((labels, job))
        print(f"Submitted [{', '.join(labels)}] -> job id {job.job_id}")

    print(f"\n{len(jobs)} job(s) submitted ({len(runs)} run(s) total). Logs: {log_dir}")
    if args.local:
        print("Waiting for local jobs to finish (this blocks; a real Slurm submission returns immediately)...")
        failures = []
        for labels, job in jobs:
            try:
                job.result()
                print(f"  done: [{', '.join(labels)}]")
            except Exception as exc:  # noqa: BLE001 - surface any job failure, then keep checking the rest
                failures.append((labels, exc))
                print(f"  FAILED: [{', '.join(labels)}] -> {exc}")
        if failures:
            raise RuntimeError(f"{len(failures)} job(s) failed: {failures}")
    else:
        print("Jobs queued on Slurm — check progress with `squeue -u $USER`, "
              "or poll job.result() from a script (blocks until that job finishes).")


if __name__ == "__main__":
    main()
