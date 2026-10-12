"""TEM-LC tuning diagnostic - run before any full 5-seed campaign.

Runs one short, gate-only (use_reward=False) training run and inspects the
lc_gating_log.npz files written at each eval checkpoint (see _flush_lc_log
in _tem_eval.py) to check the two things the gating spec explicitly calls
for: the gate should be suppressed early (network hasn't learned yet, so
most steps are "surprising") and become position-selective later (settles
into a bimodal high/low pattern tied to which states are still hard to
predict), not saturate near 1 everywhere (-> collapses to baseline Hebbian,
gate provides no information) or sit near 0.5 everywhere with no spatial
structure (-> beta too small to discriminate).

Usage
-----
    python inspect_lc_gate_tuning.py [--beta B] [--tau T] [--n-episode N] \
        [--eval-interval E] [--seed S]
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from submit_experiments import run_training_job  # noqa: E402

TMP_ROOT = os.path.join(HERE, "_lc_gate_tuning_tmp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--n-episode", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    save_root = os.path.join(TMP_ROOT, f"beta{args.beta}_tau{args.tau}")
    if os.path.exists(save_root):
        shutil.rmtree(save_root)

    overrides = {
        "TEM_SEED": str(args.seed),
        "TEM_SAVE_ROOT": save_root,
        "TEM_USE_REWARD": "0",
        "TEM_LC_GATING": "1",
        "TEM_N_EPISODE": str(args.n_episode),
        "TEM_EVAL_INTERVAL": str(args.eval_interval),
    }
    if args.beta is not None:
        overrides["TEM_LC_BETA"] = str(args.beta)
    if args.tau is not None:
        overrides["TEM_LC_EPS0_TAU"] = str(args.tau)

    print(f"Running {args.n_episode}-episode gate-only (TEM-LC) training run "
          f"(seed={args.seed}, beta={args.beta or 'default'}, tau={args.tau or 'default'})...")
    run_training_job(overrides)

    log_paths = sorted(
        glob.glob(os.path.join(save_root, "baseline_lc", "plots", "episode_*", "lc_gating_log.npz")),
        key=lambda p: int(p.split("episode_")[-1].split(os.sep)[0]),
    )
    if len(log_paths) < 2:
        print(f"FAIL - expected >=2 checkpoint logs, found {len(log_paths)} under {save_root}. "
              f"Check TEM_SAVE_MULTIENV_CSV/eval_interval settings.")
        sys.exit(1)

    def _stats(path):
        data = np.load(path)
        gate = data["gate"]
        return {
            "mean": float(np.mean(gate)),
            "std": float(np.std(gate)),
            "frac_saturated_high": float(np.mean(gate > 0.9)),
            "frac_saturated_low": float(np.mean(gate < 0.1)),
        }

    early, late = _stats(log_paths[0]), _stats(log_paths[-1])
    print(f"\nEarliest checkpoint ({log_paths[0]}):")
    print(f"  gate mean={early['mean']:.3f} std={early['std']:.3f} "
          f"frac>0.9={early['frac_saturated_high']:.3f} frac<0.1={early['frac_saturated_low']:.3f}")
    print(f"Latest checkpoint ({log_paths[-1]}):")
    print(f"  gate mean={late['mean']:.3f} std={late['std']:.3f} "
          f"frac>0.9={late['frac_saturated_high']:.3f} frac<0.1={late['frac_saturated_low']:.3f}")

    print("\nGuidance:")
    ok = True
    if early["frac_saturated_high"] > 0.5:
        print("  - Gate is already saturated near 1 EARLY (network hasn't learned yet, "
              "so this should still look surprising almost everywhere). TEM-LC will "
              "collapse to baseline Hebbian. DECREASE lc_beta and/or lc_eps0_tau.")
        ok = False
    if late["std"] < 0.05:
        print("  - Gate has very low spatial variance LATE (no selectivity emerged). "
              "INCREASE lc_beta so the gate discriminates more sharply between "
              "high- and low-surprise states.")
        ok = False
    if early["mean"] < 0.3:
        print("  - Gate is already low EARLY, before the network has learned anything - "
              "eps0 may be adapting too fast (tau too large) or beta too aggressive. "
              "DECREASE lc_eps0_tau and/or lc_beta.")
        ok = False
    if ok:
        print("  - Looks reasonable: gate not saturated early, shows spatial structure "
              "late. Proceed to the full 5-seed x 4-condition runs.")

    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
