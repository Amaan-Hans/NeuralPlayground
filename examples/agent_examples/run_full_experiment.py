"""Run the full TEM-R experiment end-to-end.

Drives the three scripts in Useful_info/how_to_run.md's "Full Run Order" so a
single command produces both trained conditions plus every plot:

1. ``whittington_2020_run.py`` with ``USE_REWARD=False``  -> results_sim/baseline/
2. ``whittington_2020_run.py`` with ``USE_REWARD=True``   -> results_sim/reward_modulated/
   (each run already calls ``_tem_eval.run_eval`` every ``eval_interval``
   episodes internally, including the final episode, so checkpoint plots
   and .npy files are produced as a side effect of training — no separate
   plotting step is needed here.)
3. ``tem_predictive_analysis.py`` -> results_sim/predictive_analysis/

Steps 1 and 2 run as two concurrent subprocesses by default (they write to
disjoint output directories, so there's no file conflict; the only shared
artifact is a diagnostic run.log whose lines may interleave between the two
processes — harmless, just noisy). Step 3 always runs after both finish, since
it reads both conditions' outputs. Pass --sequential to run 1 then 2 instead.

Usage
-----
    cd examples/agent_examples
    python run_full_experiment.py                # full 5000-episode runs, both conditions in parallel (~3h on GPU)
    python run_full_experiment.py --test          # 10-episode smoke test (~1 min total)
    python run_full_experiment.py --sequential    # baseline, then TEM-R, one at a time
    python run_full_experiment.py --skip-analysis # stop after both trainings finish
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


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
        "--test", action="store_true",
        help="10-episode smoke test (writes to results_sim_test/) instead of the full 5000-episode run.",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run baseline then TEM-R one at a time instead of concurrently.",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip tem_predictive_analysis.py after the two training runs.",
    )
    args = parser.parse_args()

    test_flag = "1" if args.test else "0"
    baseline_env = {"TEM_USE_REWARD": "0", "TEM_TEST_MODE": test_flag}
    reward_env = {"TEM_USE_REWARD": "1", "TEM_TEST_MODE": test_flag}

    if args.sequential:
        _run("whittington_2020_run.py", baseline_env)
        _run("whittington_2020_run.py", reward_env)
    else:
        _run_parallel([
            ("whittington_2020_run.py", baseline_env),
            ("whittington_2020_run.py", reward_env),
        ])

    if not args.skip_analysis:
        _run("tem_predictive_analysis.py", {"TEM_TEST_MODE": test_flag})

    print("\nAll done.")


if __name__ == "__main__":
    main()
