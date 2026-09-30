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

Defaults now match the config validated against the environment-size/rotation
audit and the cluster's proven control-landmark runs: 100-state arenas
(--arena-side 10), environment rotation ON (torch_tem's own training regime -
this is what recovers ~100% one-step predictive accuracy at matched settings,
vs ~50% without it), 10 control landmarks + decoy tiling (id 30), and the
all-envs place-cell CSV saved at every checkpoint. Pass --no-rotate /
--n-control-landmarks 0 / --decoy-object-id -1 / --no-multienv-csv to fall
back to the original NeuralPlayground behaviour piece by piece.

Usage
-----
    cd examples/agent_examples
    python run_full_experiment.py                    # full 5000-episode runs, seed 42, both conditions in parallel (~3h on GPU)
    python run_full_experiment.py --seed 123          # different seed
    python run_full_experiment.py --test              # 10-episode smoke test (~1 min total)
    python run_full_experiment.py --sequential        # baseline, then TEM-R, one at a time
    python run_full_experiment.py --skip-analysis     # stop after both trainings finish
    python run_full_experiment.py --arena-side 5      # torch_tem's own 5x5 reference size (25 states)
    python run_full_experiment.py --no-rotate         # old fixed-environment behaviour
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
    parser.add_argument(
        "--no-vary-arena-size", action="store_true",
        help="Use one uniform arena size (--arena-side) for all 16 batch slots instead of the "
             "historical 8x8/10x10/12x12-repeating mix (which is ON by default - matches the "
             "original NeuralPlayground environment-size diversity). Useful when you want size "
             "held constant as a single controlled variable.",
    )
    parser.add_argument(
        "--arena-side", type=float, default=10.0,
        help="Square arena side length shared by all 16 batch slots -> n_states = side**2 "
             "(default: 10, i.e. 100 states). Only used with --no-vary-arena-size.",
    )
    parser.add_argument(
        "--no-rotate", action="store_true",
        help="Disable environment rotation (torch_tem's own training regime - see tem_training_loop). "
             "Rotation is ON by default; this is the flag that reproduces the OLD fixed-environment behaviour.",
    )
    parser.add_argument(
        "--n-control-landmarks", type=int, default=10,
        help="Never-duplicated, reward-biased objects the value mechanism never tracks (default: 10, matches the cluster's proven config).",
    )
    parser.add_argument(
        "--decoy-object-id", type=int, default=30,
        help="Fixed object id every non-landmark state gets tiled with (default: 30). Pass -1 to restore the original per-state-random decoy layout.",
    )
    parser.add_argument(
        "--no-multienv-csv", action="store_true",
        help="Skip the all-16-environment place-cell CSV at each checkpoint (on by default).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Number of parallel environments (default: 16).",
    )
    parser.add_argument(
        "--n-episode", type=int, default=None,
        help="Override the default episode count (5000, or 10 with --test).",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=None,
        help="Override the default checkpoint interval (1000, or 2 with --test).",
    )
    args = parser.parse_args()

    test_flag = "1" if args.test else "0"
    save_root = os.path.join(
        REPO_ROOT, "experiments", "random", f"seed_{args.seed}" + ("_test" if args.test else "")
    )
    os.makedirs(save_root, exist_ok=True)

    common = {
        "TEM_TEST_MODE": test_flag,
        "TEM_SEED": str(args.seed),
        "TEM_SAVE_ROOT": save_root,
        "TEM_VARY_ARENA_SIZE": "0" if args.no_vary_arena_size else "1",
        "TEM_ARENA_SIDE": str(args.arena_side),
        "TEM_ROTATE_ENVIRONMENTS": "0" if args.no_rotate else "1",
        "TEM_N_CONTROL_LANDMARKS": str(args.n_control_landmarks),
        "TEM_DECOY_OBJECT_ID": "" if args.decoy_object_id < 0 else str(args.decoy_object_id),
        "TEM_SAVE_MULTIENV_CSV": "0" if args.no_multienv_csv else "1",
        "TEM_BATCH_SIZE": str(args.batch_size),
    }
    if args.n_episode is not None:
        common["TEM_N_EPISODE"] = str(args.n_episode)
    if args.eval_interval is not None:
        common["TEM_EVAL_INTERVAL"] = str(args.eval_interval)
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
        # Every module-level geometry global in tem_predictive_analysis.py
        # (ROOM_W/ROOM_D/XY/N_STATES/DIST_TO_REWARD) is derived at import time
        # from the historical 10x10/[3,3] default, so it must be reconfigured
        # whenever training didn't actually use that. Analysis only ever
        # looks at env 0, so what matters here is specifically env 0's size.
        if args.no_vary_arena_size:
            _side = int(args.arena_side)
            _reward_frac = 0.2
            pa.configure_geometry(
                room_w=_side, room_d=_side,
                reward_location=[args.arena_side * _reward_frac, args.arena_side * _reward_frac],
            )
        else:
            # whittington_2020_run.py's VARY_ARENA_SIZE cycle is [10, 8, 10, 12];
            # env 0 (index 0 in the cycle) is always size 10, reward fixed at [3, 3].
            pa.configure_geometry(room_w=10, room_d=10, reward_location=[3.0, 3.0])
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
