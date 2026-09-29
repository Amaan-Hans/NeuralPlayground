"""Run the full TEM-R experiment (random-policy version) end-to-end for a
given trajectory seed, saved under experiments/random_checkpoints/seed_<seed>/.

Identical to run_full_experiment.py except it launches
whittington_2020_run_checkpoints.py (which also saves an all-16-environment
place-cell CSV at episodes 1000/2000/3000/4000/5000, on top of every
existing env-0 artifact) instead of whittington_2020_run.py, and writes to
experiments/random_checkpoints/ instead of experiments/random/ - so this
never overwrites the existing seed_42/seed_123 runs under experiments/random/
(a fresh retrain won't reproduce those exactly, since torch's weight
initialization isn't seeded - only the trajectory/environment layout is,
via TRAJECTORY_SEED).

Usage
-----
    cd examples/agent_examples
    python run_full_experiment_checkpoints.py --seed 42
    python run_full_experiment_checkpoints.py --seed 123
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
        help="10-episode smoke test (writes to experiments/random_checkpoints/seed_<seed>_test/) instead of the full 5000-episode run.",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run baseline, then TEM-R one at a time instead of concurrently.",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip the post-hoc analysis after the two training runs.",
    )
    args = parser.parse_args()

    test_flag = "1" if args.test else "0"
    save_root = os.path.join(
        REPO_ROOT, "experiments", "random_checkpoints",
        f"seed_{args.seed}" + ("_test" if args.test else "")
    )
    os.makedirs(save_root, exist_ok=True)

    common = {"TEM_TEST_MODE": test_flag, "TEM_SEED": str(args.seed), "TEM_SAVE_ROOT": save_root}
    baseline_env = {**common, "TEM_USE_REWARD": "0"}
    reward_env = {**common, "TEM_USE_REWARD": "1"}

    script = "whittington_2020_run_checkpoints.py"
    if args.sequential:
        _run(script, baseline_env)
        _run(script, reward_env)
    else:
        _run_parallel([
            (script, baseline_env),
            (script, reward_env),
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
        pa.plot_landmark_activity_heatmap()
        pa.plot_value_correlation()
        pa.plot_peak_distance()
        pa.plot_grid_scores()
        pa.plot_proximal_cell_count()
        print(f"Analysis saved to: {pa.OUT_DIR}")

    print(f"\nAll done. Results in: {save_root}")


if __name__ == "__main__":
    main()
