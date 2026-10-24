"""Counterpart to probe_value_off_allenvs.py: identical fresh-probe protocol
(same episode count, warmup, periodic control-landmark rotation every
CONTROL_SHUFFLE_INTERVAL episodes), but `agent.use_reward` is left exactly
as each checkpoint was trained with - True for reward_modulated/
reward_modulated_lc, already False for baseline/baseline_lc. This is the
"value ON" counterpart under the SAME protocol as the value-OFF script, so
the only thing differing between the two datasets is the value-injection
switch itself - unlike the earlier value-ON numbers, which came from the
live-training-walk snapshot (allenvs_place_cells.csv), a different walk
protocol entirely. See probe_value_allenvs_common.py for the shared logic.

All 5 seeds x 4 conditions, one process, strictly sequential (never run
alongside any other probe - this GPU cannot share across concurrent
processes, see project_value_decodability_methodology memory).

Output columns match allenvs_place_cells.csv's schema (plus is_control_landmark):
    condition, env_id, state_id, x, y, is_landmark, is_control_landmark,
    is_reward, value, p_<freq>_<idx>...
"""
import os

from probe_value_allenvs_common import run_all

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "value_on_probe")

if __name__ == "__main__":
    run_all(OUT_DIR, suffix="valueon", force_value_off=False, overwrite=True)
