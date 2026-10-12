"""Pre-flight check for TEM-LC, required before any real 5-seed run.

Two things get verified, both against the REAL training pipeline (three
tiny whittington_2020_run.py subprocesses via submit_experiments's own
run_training_job, not a hand-rolled reimplementation):

1. Gate off reproduces baseline exactly, same seed. This is true BY
   CONSTRUCTION once Model.iteration()/Model.hebbian() are implemented
   correctly (when use_lc_gating=False, zero new tensor ops execute and
   hebbian() takes the pre-existing code path) - this check exists to catch
   an implementation mistake, not to hope the guarantee holds.
2. Gate on produces a DIFFERENT result under the same seed - catches a
   silently-inert gate (e.g. a wiring mistake that computes g(eps_t) but
   never actually multiplies it into eta).

Usage: python verify_lc_gate_off.py
"""
import os
import pickle
import shutil
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from submit_experiments import run_training_job  # noqa: E402

SEED = 4242
TMP_ROOT = os.path.join(HERE, "_lc_gate_verify_tmp")


def _run(tag: str, lc_gating: bool) -> dict:
    save_root = os.path.join(TMP_ROOT, tag)
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    run_training_job({
        "TEM_TEST_MODE": "1",  # 10 episodes, eval every 2 - fast
        "TEM_SEED": str(SEED),
        "TEM_SAVE_ROOT": save_root,
        "TEM_USE_REWARD": "0",
        "TEM_LC_GATING": "1" if lc_gating else "0",
        # Force CPU: this repo has never run torch.manual_seed before (zero
        # prior calls anywhere), and GPU reductions/backward ops are not
        # bit-deterministic by default even with a fixed seed. That's a
        # pre-existing, unrelated property of the whole stack - this check
        # is about whether gate-off reproduces exactly, so isolate it from
        # CUDA's own non-determinism by running on CPU, where PyTorch's
        # determinism guarantees actually hold.
        "CUDA_VISIBLE_DEVICES": "",
    })
    condition = "baseline_lc" if lc_gating else "baseline"
    agent_path = os.path.join(save_root, condition, "agent")
    with open(agent_path, "rb") as fh:
        return pickle.load(fh)


def _state_dicts_equal(sd_a: dict, sd_b: dict) -> bool:
    if sd_a.keys() != sd_b.keys():
        print(f"  key mismatch: {set(sd_a) ^ set(sd_b)}")
        return False
    return all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)


def main():
    print(f"Running 3 tiny training jobs (seed={SEED}, 10 episodes each)...")
    print("  [1/3] gate off (A)")
    sd_a = _run("A_gate_off", lc_gating=False)
    print("  [2/3] gate off (B, repeat)")
    sd_b = _run("B_gate_off_repeat", lc_gating=False)
    print("  [3/3] gate on (C)")
    sd_c = _run("C_gate_on", lc_gating=True)

    ok = True

    print("\nCheck 1: gate off reproduces exactly under the same seed (A == B)")
    if _state_dicts_equal(sd_a, sd_b):
        print("  PASS - every tensor in agent.tem.state_dict() is torch.equal")
    else:
        mismatched = [k for k in sd_a if not torch.equal(sd_a[k], sd_b[k])]
        print(f"  FAIL - {len(mismatched)}/{len(sd_a)} tensors differ: {mismatched[:5]}...")
        ok = False

    print("\nCheck 2: gate on differs from gate off under the same seed (A != C)")
    if not _state_dicts_equal(sd_a, sd_c):
        print("  PASS - gate-on training produced different weights (gate is live)")
    else:
        print("  FAIL - gate-on weights are IDENTICAL to gate-off - the gate is "
              "wired up but not actually affecting anything (silently inert).")
        ok = False

    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'} - "
          f"{'safe to proceed to inspect_lc_gate_tuning.py' if ok else 'fix before proceeding'}.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
