"""Predictive analysis for the loop-training experiment.

Identical analyses as tem_predictive_analysis.py but pointed at
results_sim_loop/ instead of results_sim/.

Usage
-----
    cd examples/agent_examples
    python tem_loop_predictive_analysis.py

    # For test-mode results (100 episodes, phase 2 starts at ep 51):
    python tem_loop_predictive_analysis.py --test
"""

import os
import sys

import tem_predictive_analysis as pa

# ── Override paths ─────────────────────────────────────────────────────────────
_test_mode = "--test" in sys.argv

_suffix      = "_test" if _test_mode else ""
RESULTS_ROOT = os.path.join(os.getcwd(), "results_sim_loop" + _suffix)

pa.RESULTS_ROOT = RESULTS_ROOT
pa.BASELINE_DIR = os.path.join(RESULTS_ROOT, "baseline",         "plots")
pa.REWARD_DIR   = os.path.join(RESULTS_ROOT, "reward_modulated", "plots")
pa.OUT_DIR      = os.path.join(RESULTS_ROOT, "predictive_analysis")
os.makedirs(pa.OUT_DIR, exist_ok=True)

# Loop phase starts at episode 50 (test) or 5000 (full)
pa.LOOP_START_EPISODE = 50 if _test_mode else 2500
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Results root : {RESULTS_ROOT}")
    print(f"Loop start   : ep {pa.LOOP_START_EPISODE}")
    print(f"Output       : {pa.OUT_DIR}")
    print()

    pa.plot_population_activity_maps()
    pa.plot_value_correlation()
    pa.plot_peak_distance()
    pa.plot_grid_scores()
    pa.plot_proximal_cell_count()

    print(f"\nAll plots saved to: {pa.OUT_DIR}")
