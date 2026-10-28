"""Single-pass loop probe: all 16 envs uniform 10x10, agent scripted around
a 20-state ring (border of the inner 6x6 region, states where
max(|x|,|y|)==2.5) that passes exactly through the true reward state
(reward_location=[3,3] resolves to (2.5,2.5) for every room size this
project uses - always interior, never on the true outer wall - this ring
is the innermost one that still reaches it).

4 value landmarks sit at FIXED, SPACED-OUT distances before reward (1, 3,
5, 7 ring-states away - not contiguous), plus the reward itself, plus one
ring-state right after reward that's always kept open/filler (so the
episode-250 shift always has clear space to move the whole block into,
never displacing a control). The remaining ring states are a random pool:
N_RING_CONTROLS of them hold control landmarks, the rest are plain filler -
for the 4 rotation-ON conditions this pool draw is re-randomized every
CONTROL_SHUFFLE_INTERVAL episodes (matching how those checkpoints were
actually trained); for the 4 "noshuffle" conditions it's drawn once and
left fixed. Everything off the ring (80 of each 10x10 room's 100 states) is
plain filler too. All placement is explicit, direct object-array writes -
not generate_objects()'s own random sampling.

At episode 250, the whole value block (landmarks + reward) shifts FORWARD
by one ring-state together (continuing in the direction of travel) into
the reserved-open slot.

Tracks, separately for pre (0-249) and post (250-499):
  - object-prediction accuracy by true category, per ring position (for the
    cycle plot) and overall (for the accuracy table).
  - a confusion matrix (true category x predicted category).
  - mean place-cell activity (p_inf, all frequency modules) per ring state,
    accumulated over every visit.
"""
import argparse
import importlib.util
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

AGENT_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AGENT_EXAMPLES_DIR)
sys.path.insert(0, os.path.join(AGENT_EXAMPLES_DIR, "..", ".."))
from neuralplayground.agents.whittington_2020 import Whittington2020  # noqa: E402
from tem_probe_eval_allenvs import col_names  # noqa: E402

import __main__ as _main_module  # noqa: E402


def _agent_factory(**kwargs):
    return Whittington2020(**kwargs)


_main_module._agent_factory = _agent_factory

TRAJECTORY_SEED = 42
N_EPISODES_TOTAL = 500
SHIFT_AT_EPISODE = 250
CATEGORIES = ["landmark", "control", "reward", "filler"]
BATCH_SIZE = 16
ROOM_SIZE = 10.0

_RIGHT, _UP, _LEFT, _DOWN = [1, 0], [0, 1], [-1, 0], [0, -1]
RING_SIDE = 5  # 6 distinct coordinates per axis -> 5 steps/side -> 20-state ring
# DiscreteObjectEnvironment.step() negates the y-component for pure-vertical
# actions (a[0]==0) - see _simulate_path's docstring / the original
# SQUARE_LOOP's note. So the RAW action _DOWN=[0,-1] actually moves +y, and
# _UP=[0,1] actually moves -y. Ordered here for the REAL displacement
# sequence +x, +y, -x, -y (a clean counter-clockwise loop from the
# bottom-left corner), using the raw labels that produce that.
RING_LOOP = [_RIGHT] * RING_SIDE + [_DOWN] * RING_SIDE + [_LEFT] * RING_SIDE + [_UP] * RING_SIDE
LOOP_START = (-2.5, -2.5)
REWARD_RING_IDX = 10    # ring index 10 = (2.5, 2.5) = true reward state
N_RING_LANDMARKS = 5    # ids 0-4, at FIXED, SPACED-OUT distances before reward (not contiguous) - id 0
                        # furthest, id 4 closest. The gaps between them are part of the general random pool.
LANDMARK_DISTANCES = [9, 7, 5, 3, 1]  # landmark i sits at ring index (reward_idx - LANDMARK_DISTANCES[i])
RESERVED_OPEN_OFFSET = 1  # ring index (reward_idx + 1) is always kept filler/empty - never a control - so
                          # the episode-250 shift (whole block +1 ring-index) always has clear space to move into.
N_RING_CONTROLS = 5     # ids 10-14, placed on a random 5 of the remaining non-reserved ring states - the
                        # rest stay filler. Re-randomized every CONTROL_SHUFFLE_INTERVAL episodes for
                        # rotation-ON conditions; fixed for the rotation-OFF ("noshuffle") ones.
CONTROL_SHUFFLE_INTERVAL = 25


def value_block(reward_idx, n_ring_states):
    """The reserved ring indices for a given reward position: 4 landmark
    slots at fixed distances before it, the reward itself, and the always-
    open slot right after it. Everything else is the random control/filler
    pool. Shifting the whole experiment forward by 1 is just calling this
    with reward_idx+1 - the relative structure (spacing + open slot)
    carries over unchanged since every offset is relative to reward_idx."""
    landmark_indices = [(reward_idx - d) % n_ring_states for d in LANDMARK_DISTANCES]
    reserved_open_idx = (reward_idx + RESERVED_OPEN_OFFSET) % n_ring_states
    return landmark_indices, reward_idx % n_ring_states, reserved_open_idx


def _simulate_path(start, actions, step_size=1):
    path = []
    pos = np.array(start, dtype=float)
    for a in actions:
        a = np.array(a, dtype=float)
        a_rev = np.array([0.0, -a[1]]) if a[0] == 0 else a
        norm = np.linalg.norm(a_rev)
        if norm > 0:
            a_rev = a_rev / norm
        pos = pos + step_size * a_rev
        path.append(tuple(pos))
    return path


def _load_model_from_save(save_path):
    model_file = os.path.join(save_path, "whittington_2020_model.py")
    spec = importlib.util.spec_from_file_location("tem_model_saved_" + save_path, model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model


def build_ring_env_agent(save_path):
    training_dict = pd.read_pickle(os.path.join(save_path, "params.dict"))
    agent_params = dict(training_dict["agent_params"])
    env_params = dict(training_dict["env_params"])
    discrete_env_params = dict(env_params["arg_env_params"])
    state_dict = pd.read_pickle(os.path.join(save_path, "agent"))
    _load_model_from_save(save_path)

    # Force all BATCH_SIZE envs to uniform ROOM_SIZE x ROOM_SIZE.
    env_params["batch_size"] = BATCH_SIZE
    env_params["arena_x_limits"] = [[-ROOM_SIZE / 2, ROOM_SIZE / 2]] * BATCH_SIZE
    env_params["arena_y_limits"] = [[-ROOM_SIZE / 2, ROOM_SIZE / 2]] * BATCH_SIZE
    env_params["arg_env_params"] = discrete_env_params
    agent_params["batch_size"] = BATCH_SIZE
    agent_params["room_widths"] = [ROOM_SIZE] * BATCH_SIZE
    agent_params["room_depths"] = [ROOM_SIZE] * BATCH_SIZE
    agent_params["state_densities"] = [1] * BATCH_SIZE

    full_path = [LOOP_START] + _simulate_path(LOOP_START, RING_LOOP, step_size=1)
    assert np.allclose(full_path[-1], full_path[0]), f"ring not closed: {full_path[0]} vs {full_path[-1]}"
    unique_path_states = full_path[:-1]
    n_ring_states = len(unique_path_states)
    assert np.allclose(unique_path_states[REWARD_RING_IDX], (2.5, 2.5)), (
        f"expected ring index {REWARD_RING_IDX} to be the reward corner (2.5, 2.5), "
        f"got {unique_path_states[REWARD_RING_IDX]}"
    )

    # 4 landmarks at fixed, spaced-out distances before reward (not
    # contiguous - the gaps between them are part of the general random
    # control/filler pool), plus the reward itself and an always-open slot
    # right after it (reserved so the shift always has clear space to move
    # into, never needing to displace a control).
    landmark_idx, reward_idx, open_idx = value_block(REWARD_RING_IDX, n_ring_states)
    landmark_states = [unique_path_states[i] for i in landmark_idx]
    reward_xy = unique_path_states[reward_idx]
    value_block_indices = set(landmark_idx) | {reward_idx, open_idx}
    non_value_indices = [i for i in range(n_ring_states) if i not in value_block_indices]
    non_value_states = [unique_path_states[i] for i in non_value_indices]

    # Shift FORWARD by 1 (continuing in the direction of travel): the whole
    # (landmarks + reward) block moves into the always-open slot and beyond
    # - since every offset in value_block() is relative to reward_idx, this
    # is just calling it again with reward_idx+1; the relative spacing and
    # the new open slot both carry over automatically.
    new_landmark_idx, new_reward_idx, new_open_idx = value_block(REWARD_RING_IDX + 1, n_ring_states)
    shifted_landmark_states = [unique_path_states[i] for i in new_landmark_idx]
    new_reward_xy = unique_path_states[new_reward_idx]
    new_value_block_indices = set(new_landmark_idx) | {new_reward_idx, new_open_idx}
    # Post-shift's OWN non-value pool - a rotation-ON reshuffle that happens
    # after the shift must draw from this, not the pre-shift pool (which
    # still includes the old landmark/reward/open positions, now wrong).
    non_value_states_post = [
        unique_path_states[i] for i in range(n_ring_states) if i not in new_value_block_indices
    ]

    env = training_dict["env_class"](**env_params)
    agent = training_dict["agent_class"](**agent_params)
    agent.tem.load_state_dict(state_dict)
    agent.tem.eval()

    n_landmarks = discrete_env_params.get("n_landmarks", 10)
    n_control_landmarks = discrete_env_params.get("n_control_landmarks", 0)
    n_special = n_landmarks + n_control_landmarks
    reward_object_id = env.environments[0].reward_object_id

    return {
        "env": env, "agent": agent, "unique_path_states": unique_path_states, "n_ring_states": n_ring_states,
        "landmark_states": landmark_states, "reward_xy": reward_xy, "non_value_states": non_value_states,
        "non_value_states_post": non_value_states_post, "open_xy": unique_path_states[open_idx],
        "shifted_landmark_states": shifted_landmark_states, "new_reward_xy": new_reward_xy,
        "n_landmarks": n_landmarks, "n_control_landmarks": n_control_landmarks, "n_special": n_special,
        "reward_object_id": reward_object_id,
    }


def pick_control_states(pool, rng):
    """Randomly choose N_RING_CONTROLS of the given (non-landmark,
    non-reward, non-open) ring-state pool to hold control ids this cycle -
    the rest stay filler. `pool` must be cfg["non_value_states"] (pre-shift
    block) or cfg["non_value_states_post"] (post-shift block) - NOT always
    the pre-shift one, since after the shift the old pool includes states
    that are now landmarks/reward/the open slot. Uses the stdlib `random`
    module (the `rng` passed in is the `random` module itself or a
    random.Random instance) so it's reproducible under random.seed(),
    matching DiscreteObjectEnvironment.reshuffle_control_landmarks()'s own
    convention."""
    return rng.sample(pool, N_RING_CONTROLS)


def place_ring_objects(env, cfg, control_states):
    """Explicit landmark/reward/control placement, every env slot identical.
    MUST be called AFTER env.reset() - DiscreteObjectEnvironment.reset()
    unconditionally calls self.objects = self.generate_objects() (random
    layout), so anything written before reset() gets silently wiped out."""
    landmark_states = cfg["landmark_states"]
    reward_xy = cfg["reward_xy"]
    n_landmarks = cfg["n_landmarks"]
    reward_object_id = cfg["reward_object_id"]

    for j in range(BATCH_SIZE):
        env_j = env.environments[j]
        poss_objects = np.eye(env_j.n_objects)
        # Everything off the value block starts as filler/decoy (including
        # the reserved always-open slot, which never gets a control written
        # on top of it); control ids then get written over a random subset
        # of the non-value pool below.
        for xy in cfg["non_value_states"] + [cfg["open_xy"]]:
            sid = int(env_j.pos_to_state(np.array(xy)))
            if env_j.decoy_object_id is not None:
                env_j.objects[sid, :] = poss_objects[env_j.decoy_object_id]
        for i, xy in enumerate(landmark_states):
            sid = int(env_j.pos_to_state(np.array(xy)))
            env_j.objects[sid, :] = poss_objects[i]
        if reward_object_id is not None:
            reward_sid = int(env_j.pos_to_state(np.array(reward_xy)))
            env_j.objects[reward_sid, :] = poss_objects[reward_object_id]
        for i, xy in enumerate(control_states):
            sid = int(env_j.pos_to_state(np.array(xy)))
            env_j.objects[sid, :] = poss_objects[n_landmarks + i]


def reshuffle_ring_controls(env, cfg, old_control_states, pool, rng):
    """Vacate the OLD control states back to filler/decoy, then draw a FRESH
    random set of N_RING_CONTROLS states from `pool` and write control ids
    there - the rotation-ON conditions' per-25-episode analogue of
    reshuffle_control_landmarks(), kept ring-local (not room-wide) so the
    scripted loop still actually visits them. `pool` must match whichever
    block (pre- or post-shift) is currently active."""
    new_control_states = pick_control_states(pool, rng)
    n_landmarks = cfg["n_landmarks"]

    for j in range(BATCH_SIZE):
        env_j = env.environments[j]
        poss_objects = np.eye(env_j.n_objects)
        for xy in old_control_states:
            sid = int(env_j.pos_to_state(np.array(xy)))
            if env_j.decoy_object_id is not None:
                env_j.objects[sid, :] = poss_objects[env_j.decoy_object_id]
        for i, xy in enumerate(new_control_states):
            sid = int(env_j.pos_to_state(np.array(xy)))
            env_j.objects[sid, :] = poss_objects[n_landmarks + i]
    return new_control_states


def apply_shift(env, agent, cfg):
    # Vacate the FULL old block (10 landmarks + the reward state, 11
    # distinct states), then write the full NEW block (10 shifted
    # landmarks + the new reward state at its own separate position) -
    # there's room for both distinctly here, unlike the old 14-state loop
    # design where reward had to double up with the last landmark slot.
    old_positions = list(cfg["landmark_states"]) + [cfg["reward_xy"]]
    new_landmark_positions = cfg["shifted_landmark_states"]
    new_reward_xy = cfg["new_reward_xy"]
    n_special = cfg["n_special"]

    for j in range(BATCH_SIZE):
        env_j = env.environments[j]
        poss_objects = np.eye(env_j.n_objects)
        for xy in old_positions:
            sid = int(env_j.pos_to_state(np.array(xy)))
            if env_j.decoy_object_id is not None:
                env_j.objects[sid, :] = poss_objects[env_j.decoy_object_id]
            else:
                rand = random.randint(n_special, env_j.n_objects - 1)
                while rand == env_j.reward_object_id:
                    rand = random.randint(n_special, env_j.n_objects - 1)
                env_j.objects[sid, :] = poss_objects[rand]
        for i, xy in enumerate(new_landmark_positions):
            sid = int(env_j.pos_to_state(np.array(xy)))
            env_j.objects[sid, :] = poss_objects[i]
        if env_j.reward_object_id is not None:
            reward_sid = int(env_j.pos_to_state(np.array(new_reward_xy)))
            env_j.objects[reward_sid, :] = poss_objects[env_j.reward_object_id]
        env_j.reward_location = list(new_reward_xy)
        agent.reward_state_ids[j] = int(env_j.pos_to_state(np.array(new_reward_xy)))


def snapshot_ring_categories(env, cfg, n_landmarks, n_special, reward_object_id):
    """Read the TRUE current category at every ring position directly from
    env slot 0's object array - used instead of inferring it analytically,
    so the place-cell CSV's is_landmark/is_control/is_reward flags can
    never drift out of sync with whatever apply_shift actually wrote."""
    env0 = env.environments[0]
    cats = []
    for xy in cfg["unique_path_states"]:
        sid = int(env0.pos_to_state(np.array(xy)))
        obj_id = int(np.argmax(env0.objects[sid]))
        cats.append(categorize(obj_id, n_landmarks, n_special, reward_object_id))
    return cats


def categorize(obj_id, n_landmarks, n_special, reward_object_id):
    if reward_object_id is not None and obj_id == reward_object_id:
        return "reward"
    if obj_id < n_landmarks:
        return "landmark"
    if obj_id < n_special:
        return "control"
    return "filler"


def run_probe(save_path, label, out_dir, rotation_enabled):
    cfg = build_ring_env_agent(save_path)
    env, agent = cfg["env"], cfg["agent"]
    n_envs = BATCH_SIZE
    n_rollout = agent.pars["n_rollout"]
    n_ring_states = cfg["n_ring_states"]
    n_landmarks, n_special = cfg["n_landmarks"], cfg["n_special"]
    reward_object_id = cfg["reward_object_id"]
    n_p_list = agent.pars["n_p"]
    p_cols = col_names(n_p_list)

    print(f"[{label}] ring={n_ring_states} states, {N_RING_LANDMARKS} landmarks, {N_RING_CONTROLS} controls "
          f"(of {len(cfg['non_value_states'])} non-value slots), reward at ring idx {REWARD_RING_IDX}, "
          f"rotation_enabled={rotation_enabled}, shift at episode {SHIFT_AT_EPISODE}", flush=True)

    random.seed(TRAJECTORY_SEED)
    np.random.seed(TRAJECTORY_SEED)
    obs, state = env.reset(random_state=False, custom_state=list(LOOP_START))
    control_states = pick_control_states(cfg["non_value_states"], random)
    place_ring_objects(env, cfg, control_states)
    # reset()'s own obs reflects the pre-placement random layout (it calls
    # generate_objects() as its last act, before place_ring_objects() runs)
    # - recompute it now that the real ring layout is in place, same [index,
    # object, pos] format make_object_observation always returns.
    obs = [env.environments[j].make_object_observation(np.array(LOOP_START)) for j in range(BATCH_SIZE)]
    agent.n_walk = 0
    agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

    loop_idx = 0
    prev_iter = None
    rows = []
    confusion = {half: {t: {p: 0 for p in CATEGORIES} for t in CATEGORIES} for half in ("pre", "post")}
    pos_accum = {
        half: [{c: {"correct": 0, "total": 0} for c in CATEGORIES} for _ in range(n_ring_states)]
        for half in ("pre", "post")
    }
    # place-cell accumulation per (env, ring state), per half: running sum +
    # count of p_inf (all modules concatenated). Kept PER ENV (not pooled
    # across envs) so the has_value classifier downstream still has a
    # proper (seed, env_id, state) population to hold out by, instead of
    # collapsing to one pooled sample per position per half.
    p_sum = {half: np.zeros((n_envs, n_ring_states, sum(n_p_list))) for half in ("pre", "post")}
    p_count = {half: np.zeros((n_envs, n_ring_states), dtype=np.int64) for half in ("pre", "post")}
    ring_categories = {"pre": snapshot_ring_categories(env, cfg, n_landmarks, n_special, reward_object_id), "post": None}

    with torch.no_grad():
        for episode in range(N_EPISODES_TOTAL):
            half = "pre" if episode < SHIFT_AT_EPISODE else "post"
            loop_idx_log = []
            while agent.n_walk < n_rollout:
                loop_idx_log.append(loop_idx % n_ring_states)
                actions = agent.batch_act(obs)
                loop_action = RING_LOOP[loop_idx % len(RING_LOOP)]
                env0_state_pre = obs[0][0]
                for i in range(n_envs):
                    actions[i] = loop_action
                    agent.prev_actions[i] = loop_action
                obs, state, reward = env.step(actions, normalize_step=True)
                if obs[0][0] != env0_state_pre:
                    loop_idx += 1
            agent.n_walk = 0

            real_indices = [i for i, step in enumerate(agent.obs_history) if step[0][0] != -1]
            history = [agent.obs_history[i] for i in real_indices]
            actions_raw = [agent.walk_actions[i] for i in real_indices]
            held_raw = [agent.held_landmark_history[i] for i in real_indices]
            # -1 correction: batch_act() appends a COPY of self.prev_observations
            # to obs_history BEFORE updating it to the obs passed in this call -
            # so the obs_history entry recorded during iteration k actually holds
            # iteration (k-1)'s position, one behind loop_idx_log[k] (logged at
            # the top of iteration k, after iteration (k-1)'s increment already
            # happened). Confirmed by direct trace: without this, object content
            # at a given labeled ring_pos didn't match what was actually placed
            # there.
            pos_at_step = [(loop_idx_log[i] - 1) % n_ring_states for i in real_indices]
            chunk_len = len(history)
            if chunk_len == 0:
                agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []
                continue

            locations = [[{"id": step[j][0], "shiny": None} for j in range(n_envs)] for step in history]
            observations = [[step[j][1] for j in range(n_envs)] for step in history]
            action_values = agent.step_to_actions(actions_raw)
            obs_array = np.reshape(observations, (chunk_len, n_envs, agent.pars["n_x"]))
            action_array = np.reshape(action_values, (chunk_len, n_envs))

            v_seq = None
            if agent.use_reward and agent.td is not None:
                v_seq = agent._value_for_history(held_raw)

            model_input = []
            for i in range(chunk_len):
                step = [locations[i], torch.from_numpy(obs_array[i]).type(torch.float32).to(agent.device),
                        action_array[i].tolist()]
                if v_seq is not None:
                    step.append(None)
                    step.append(v_seq[i])
                model_input.append(step)

            forward = agent.tem(model_input, prev_iter)

            cat_correct = {c: [] for c in CATEGORIES}
            for local_i, step in enumerate(forward):
                x_actual = step.x.detach().cpu().numpy()
                x_pred = step.x_gen[2].detach().cpu().numpy()
                true_ids = np.argmax(x_actual, axis=-1)
                pred_ids = np.argmax(x_pred, axis=-1)
                pos = pos_at_step[local_i]
                p_vec_all = np.concatenate([step.p_inf[f].detach().cpu().numpy() for f in range(len(n_p_list))], axis=-1)
                for j in range(n_envs):
                    true_cat = categorize(int(true_ids[j]), n_landmarks, n_special, reward_object_id)
                    pred_cat = categorize(int(pred_ids[j]), n_landmarks, n_special, reward_object_id)
                    correct = true_cat == pred_cat
                    cat_correct[true_cat].append(correct)
                    confusion[half][true_cat][pred_cat] += 1
                    pos_accum[half][pos][true_cat]["total"] += 1
                    pos_accum[half][pos][true_cat]["correct"] += int(correct)
                    p_sum[half][j, pos] += p_vec_all[j]
                    p_count[half][j, pos] += 1

            rows.append({
                "episode": episode, "half": half,
                **{f"acc_{c}": (float(np.mean(cat_correct[c])) if cat_correct[c] else float("nan")) for c in CATEGORIES},
                **{f"n_{c}": len(cat_correct[c]) for c in CATEGORIES},
            })

            prev_iter = [forward[-1]]
            agent.obs_history, agent.walk_actions, agent.held_landmark_history = [], [], []

            if episode + 1 == SHIFT_AT_EPISODE:
                print(f"[{label}]   >>> SHIFT at episode {episode + 1} <<<", flush=True)
                apply_shift(env, agent, cfg)
                ring_categories["post"] = snapshot_ring_categories(env, cfg, n_landmarks, n_special, reward_object_id)
                # The shift may have overwritten one or more currently-tracked
                # control positions with a new landmark/reward (the block's
                # leading edge can land on old pool territory) - drop those
                # from our tracking list so a later reshuffle's "vacate old
                # controls" step never clobbers a real landmark. For
                # rotation-OFF conditions this permanently reduces the active
                # control count by however many got absorbed (no replacement
                # drawn, matching "never reshuffled"); for rotation-ON it's
                # moot past the next reshuffle anyway.
                new_value_positions = set(cfg["shifted_landmark_states"]) | {cfg["new_reward_xy"]}
                control_states = [xy for xy in control_states if xy not in new_value_positions]

            if rotation_enabled and (episode + 1) % CONTROL_SHUFFLE_INTERVAL == 0:
                pool = cfg["non_value_states_post"] if episode + 1 >= SHIFT_AT_EPISODE else cfg["non_value_states"]
                control_states = reshuffle_ring_controls(env, cfg, control_states, pool, random)
                if episode + 1 >= SHIFT_AT_EPISODE:
                    ring_categories["post"] = snapshot_ring_categories(env, cfg, n_landmarks, n_special, reward_object_id)
                else:
                    ring_categories["pre"] = snapshot_ring_categories(env, cfg, n_landmarks, n_special, reward_object_id)

            if (episode + 1) % 100 == 0:
                print(f"  [{label}] episode {episode + 1}/{N_EPISODES_TOTAL}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{label}_episodes.csv"), index=False)
    for half in ("pre", "post"):
        cm = pd.DataFrame(confusion[half]).T.reindex(index=CATEGORIES, columns=CATEGORIES)
        cm.to_csv(os.path.join(out_dir, f"{label}_confusion_{half}.csv"))

    pos_rows = []
    for half in ("pre", "post"):
        for pos in range(n_ring_states):
            for c in CATEGORIES:
                d = pos_accum[half][pos][c]
                pos_rows.append({"half": half, "ring_pos": pos, "category": c,
                                  "correct": d["correct"], "total": d["total"],
                                  "accuracy": (d["correct"] / d["total"]) if d["total"] else float("nan")})
    pd.DataFrame(pos_rows).to_csv(os.path.join(out_dir, f"{label}_position_accuracy.csv"), index=False)

    # Place-cell CSV (one row per ring state per half) - same schema style as
    # allenvs_place_cells.csv so the existing has_value recipe applies directly.
    # is_landmark/is_control_landmark/is_reward come from the real snapshot
    # taken at each half's own boundary (snapshot_ring_categories), not an
    # inferred/guessed position formula.
    place_rows = []
    for half in ("pre", "post"):
        for j in range(n_envs):
            for pos in range(n_ring_states):
                if p_count[half][j, pos] == 0:
                    continue
                xy = cfg["unique_path_states"][pos]
                mean_p = p_sum[half][j, pos] / p_count[half][j, pos]
                cat = ring_categories[half][pos]
                row = {
                    "half": half, "env_id": j, "ring_pos": pos, "state_id": pos, "x": xy[0], "y": xy[1],
                    "is_landmark": int(cat == "landmark"),
                    "is_control_landmark": int(cat == "control"),
                    "is_reward": int(cat == "reward"),
                    "n_visits": int(p_count[half][j, pos]),
                }
                row.update({c: v for c, v in zip(p_cols, mean_p)})
                place_rows.append(row)
    pd.DataFrame(place_rows).to_csv(os.path.join(out_dir, f"{label}_place_cells.csv"), index=False)

    print(f"[{label}] saved episodes, confusion, position_accuracy, place_cells to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rotation-enabled", action="store_true",
                         help="Reshuffle controls every CONTROL_SHUFFLE_INTERVAL episodes (rotation-ON conditions).")
    args = parser.parse_args()
    run_probe(args.save_path, args.label, args.out_dir, args.rotation_enabled)


if __name__ == "__main__":
    main()
