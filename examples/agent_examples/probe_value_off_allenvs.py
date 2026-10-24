"""Isolate whether the landmark-reward-distance decodability found for
reward_modulated/reward_modulated_lc (check_landmark_reward_predictiveness.py)
comes from the TRAINED WEIGHTS (learned under a value objective) or from the
TD value signal being actively injected into x_c at probe time.

Forces `agent.use_reward = False` after loading each checkpoint's trained
weights - value injection into model_input is suppressed regardless of what
the checkpoint was trained with, while the weights themselves (and the
periodic control-landmark rotation every CONTROL_SHUFFLE_INTERVAL episodes,
matching training) stay identical to probe_value_on_allenvs.py's run. See
probe_value_allenvs_common.py for the shared probing logic both scripts use.

All 5 seeds x 4 conditions, one process, strictly sequential (never run
alongside any other probe - this GPU cannot share across concurrent
processes, see project_value_decodability_methodology memory).

Output columns match allenvs_place_cells.csv's schema (plus is_control_landmark):
    condition, env_id, state_id, x, y, is_landmark, is_control_landmark,
    is_reward, value, p_<freq>_<idx>...
"""
import os

from probe_value_allenvs_common import run_all

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "value_off_probe")

if __name__ == "__main__":
    run_all(OUT_DIR, suffix="valueoff", force_value_off=True, overwrite=True)
