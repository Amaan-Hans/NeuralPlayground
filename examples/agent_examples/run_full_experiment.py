"""Run the full TEM-R experiment (random-policy version) end-to-end for a
given trajectory seed, saved under experiments/random/seed_<seed>/.

Drives the two training runs plus the post-hoc analysis, mirroring
run_loop_experiment.py's driver pattern but for whittington_2020_run.py's
pure-random-policy script instead of the loop (explore-then-square-loop)
script:

1. ``whittington_2020_run.py`` with ``USE_REWARD=False``  -> <root>/baseline/
2. ``whittington_2020_run.py`` with ``USE_REWARD=True``   -> <root>/reward_modulated/
   (each run already calls ``_tem_eval.run_eval`` every ``eval_interval``
   episodes internally, including episode 1 and the final episode, so
   checkpoint plots and .npy files are produced as a side effect of
   training — no separate plotting step is needed here.)
3. Post-hoc analysis (imported in-process, same functions
   ``tem_predictive_analysis.py`` exposes) -> <root>/predictive_analysis/

Steps 1 and 2 run as two concurrent subprocesses by default (they write to
disjoint output directories, so there's no file conflict; the only shared
artifact is a diagnostic run.log whose lines may interleave between the two
processes — harmless, just noisy). Step 3 always runs after both finish, since
it reads both conditions' outputs. Pass --sequential to run 1 then 2 instead.

Usage
-----
    cd examples/agent_examples
    python run_full_experiment.py                 # full 5000-episode runs, seed 42, both conditions in parallel (~3h on GPU)
    python run_full_experiment.py --seed 123       # different seed
    python run_full_experiment.py --test           # 10-episode smoke test (~1 min total)
    python run_full_experiment.py --sequential     # baseline, then TEM-R, one at a time
    python run_full_experiment.py --skip-analysis  # stop after both trainings finish
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def _start(script: str, env_overrides: dict) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(env_overrides)
    print(f"\n{'=' * 70}\nStarting {script}  {env_overrides}\n{'=' * 70}", flush=True)
    return subprocess.Popen([sys.executable, script], cwd=HERE, env=env)


def _run(script: str, env_overrides: dict):
    """Start a script and block until it finishes, raising on failure."""
    proc = _start(script, env_overrides)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"{script} exited with code {returncode} (env={env_overrides})")


def _run_parallel(jobs: list):
    """Start every (script, env_overrides) job concurrently, then wait for all."""
    procs = [(_start(script, env), script, env) for script, env in jobs]
    failures = []
    for proc, script, env in procs:
        returncode = proc.wait()
        if returncode != 0:
            failures.append(f"{script} exited with code {returncode} (env={env})")
    if failures:
        raise RuntimeError("One or more training runs failed:\n" + "\n".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="TRAJECTORY_SEED shared by both conditions (default: 42).",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="10-episode smoke test (writes to experiments/random/seed_<seed>_test/) instead of the full 5000-episode run.",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run baseline then TEM-R one at a time instead of concurrently.",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip the post-hoc analysis after the two training runs.",
    )
    args = parser.parse_args()

    test_flag = "1" if args.test else "0"
    save_root = os.path.join(
        REPO_ROOT, "experiments", "random", f"seed_{args.seed}" + ("_test" if args.test else "")
    )
    os.makedirs(save_root, exist_ok=True)

    common = {"TEM_TEST_MODE": test_flag, "TEM_SEED": str(args.seed), "TEM_SAVE_ROOT": save_root}
    baseline_env = {**common, "TEM_USE_REWARD": "0"}
    reward_env = {**common, "TEM_USE_REWARD": "1"}

    if args.sequential:
        _run("whittington_2020_run.py", baseline_env)
        _run("whittington_2020_run.py", reward_env)
    else:
        _run_parallel([
            ("whittington_2020_run.py", baseline_env),
            ("whittington_2020_run.py", reward_env),
        ])

    if not args.skip_analysis:
        baseline_plots = os.path.join(save_root, "baseline", "plots")
        reward_plots = os.path.join(save_root, "reward_modulated", "plots")

        print("\nBoth conditions done — running predictive analysis...")
        sys.path.insert(0, HERE)
        import tem_predictive_analysis as pa

        pa.RESULTS_ROOT = save_root
        pa.BASELINE_DIR = baseline_plots
        pa.REWARD_DIR = reward_plots
        pa.OUT_DIR = os.path.join(save_root, "predictive_analysis")
        os.makedirs(pa.OUT_DIR, exist_ok=True)
        pa.plot_population_activity_maps()
        pa.plot_value_correlation()
        pa.plot_peak_distance()
        pa.plot_grid_scores()
        pa.plot_proximal_cell_count()
        print(f"Analysis saved to: {pa.OUT_DIR}")

    print(f"\nAll done. Results in: {save_root}")


if __name__ == "__main__":
    main()
