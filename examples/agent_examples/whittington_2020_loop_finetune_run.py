"""Temporary experiment: take an already-trained checkpoint (any of the 8
seed_42 conditions from the ctrlshuffle25_rewardobj campaign) and continue
training it - real gradient + Hebbian + TD updates via agent.update(), not an
eval/no_grad probe - on the SAME 20-state ring-loop structure
probe_loop_ring.py already uses to evaluate these checkpoints (5 landmarks at
fixed distances 9/7/5/3/1 from reward, reward itself, a reserved-open slot,
and a random control/filler pool that reshuffles every
CONTROL_SHUFFLE_INTERVAL episodes for the 4 rotation-ON conditions).

Runs TEM_N_EPISODES total (default 1000): the value block (landmarks +
reward) shifts forward by 1 ring-state at TEM_SHIFT_AT_EPISODE (default
500), exactly like the probe's own shift. A checkpoint is saved into
<save_path>/pre_shift_checkpoint/ right before the shift is applied (so the
"about to be fine-tuned into the shift" state is never lost), then training
continues through the shift for the remaining episodes, ending with a final
checkpoint saved into <save_path>/ itself.

Loads the trained WEIGHTS only (agent.tem.load_state_dict) - a fresh agent/
optimizer/Hebbian-memory/TD-table is constructed around them, matching how
this repo's existing --load-checkpoint curriculum-learning path already
works. The annealing schedule (eta/lambda/lr/loss-weight ramp, tuned for the
checkpoint's original 50000-episode training) is RESCALED to this run's own
short TEM_N_EPISODES length, so it fully ramps by the end of this fine-tune
instead of running a tiny, barely-annealed slice of the original 50k
schedule - same convention every other entry script in this repo uses
(parameters.rescale_schedule_for_train_it), just reapplied on top of the
checkpoint's already-saved (50k-scaled) params; the two rescalings compose
correctly since rescale_schedule_for_train_it's scale factor is multiplicative
each time.

Never writes into TEM_BASE_CHECKPOINT_DIR - only ever reads from it.

Env vars
--------
TEM_BASE_CHECKPOINT_DIR : path to the source seed_42/<condition> checkpoint
                          to load weights from (required).
TEM_SAVE_ROOT           : output directory for this fine-tune's checkpoints
                          (required) - must not already exist with content,
                          so this can never silently overwrite a prior run.
TEM_SEED                : trajectory seed for the ring walk / control-shuffle
                          RNG (default 42) - NOT the checkpoint's own training
                          seed, which is fixed by whichever checkpoint is
                          loaded; this just controls this fine-tune's own
                          scripted-walk/reshuffle randomness.
TEM_ROTATION_ENABLED   : "1"/"0" - reshuffle ring controls every
                          CONTROL_SHUFFLE_INTERVAL episodes (default "1").
TEM_N_EPISODES          : total fine-tune episodes (default 1000).
TEM_SHIFT_AT_EPISODE    : episode after which the pre-shift checkpoint is
                          saved and the value-block shift is applied
                          (default 500).
"""
import os
import pickle
import random
import shutil
import sys

import numpy as np

AGENT_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AGENT_EXAMPLES_DIR)
from probe_loop_ring import (  # noqa: E402
    BATCH_SIZE, CONTROL_SHUFFLE_INTERVAL, LOOP_START, RING_LOOP,
    apply_shift, build_ring_env_agent, pick_control_states, place_ring_objects,
    reshuffle_ring_controls,
)
from neuralplayground.agents.whittington_2020_extras import (  # noqa: E402
    whittington_2020_parameters as parameters,
)

BASE_CHECKPOINT_DIR = os.environ["TEM_BASE_CHECKPOINT_DIR"]
SAVE_ROOT = os.environ["TEM_SAVE_ROOT"]
TRAJECTORY_SEED = int(os.environ.get("TEM_SEED", "42"))
ROTATION_ENABLED = os.environ.get("TEM_ROTATION_ENABLED", "1") == "1"
N_EPISODES_TOTAL = int(os.environ.get("TEM_N_EPISODES", "1000"))
SHIFT_AT_EPISODE = int(os.environ.get("TEM_SHIFT_AT_EPISODE", "500"))


def save_checkpoint(agent, out_dir, src_model_file, training_dict):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "agent"), "wb") as f:
        pickle.dump(agent.tem.state_dict(), f)
    with open(os.path.join(out_dir, "agent_hyper"), "wb") as f:
        pickle.dump(agent.tem.hyper, f)
    with open(os.path.join(out_dir, "params.dict"), "wb") as f:
        pickle.dump(training_dict, f)
    shutil.copy(src_model_file, os.path.join(out_dir, "whittington_2020_model.py"))


def main():
    if os.path.isdir(SAVE_ROOT) and os.listdir(SAVE_ROOT):
        raise RuntimeError(
            f"TEM_SAVE_ROOT already has content: {SAVE_ROOT} - refusing to risk overwriting "
            f"a prior fine-tune run. Use a fresh directory."
        )
    os.makedirs(SAVE_ROOT, exist_ok=True)

    print(f"Loading base checkpoint from: {BASE_CHECKPOINT_DIR}", flush=True)
    cfg = build_ring_env_agent(BASE_CHECKPOINT_DIR)
    env, agent = cfg["env"], cfg["agent"]
    # build_ring_env_agent() calls agent.tem.eval() (it's written for
    # probe_loop_ring.py's own read-only forward passes) - undo that here:
    # this script does real gradient training, and with eval() left on,
    # update()'s loss carries no grad_fn (observed: "element 0 of tensors
    # does not require grad") since whichever branch the saved model takes
    # for self.training=False isn't meant to be backpropagated through.
    agent.tem.train()
    n_envs = BATCH_SIZE
    n_rollout = agent.pars["n_rollout"]

    # Rescale the annealing schedule to THIS fine-tune's own (short) length,
    # composing correctly on top of whatever scaling the checkpoint's original
    # 50000-episode training already baked in (see module docstring).
    agent.pars = parameters.rescale_schedule_for_train_it(agent.pars, N_EPISODES_TOTAL)

    src_model_file = os.path.join(BASE_CHECKPOINT_DIR, "whittington_2020_model.py")
    import pandas as pd
    training_dict = pd.read_pickle(os.path.join(BASE_CHECKPOINT_DIR, "params.dict"))

    print(
        f"base={BASE_CHECKPOINT_DIR} save_root={SAVE_ROOT} seed={TRAJECTORY_SEED} "
        f"rotation_enabled={ROTATION_ENABLED} n_episodes={N_EPISODES_TOTAL} "
        f"shift_at_episode={SHIFT_AT_EPISODE} n_rollout={n_rollout}",
        flush=True,
    )

    random.seed(TRAJECTORY_SEED)
    np.random.seed(TRAJECTORY_SEED)
    obs, state = env.reset(random_state=False, custom_state=list(LOOP_START))
    control_states = pick_control_states(cfg["non_value_states"], random)
    place_ring_objects(env, cfg, control_states)
    obs = [env.environments[j].make_object_observation(np.array(LOOP_START)) for j in range(BATCH_SIZE)]
    # Deliberately NOT resetting agent.n_walk here - it starts at -1 from
    # Whittington2020.__init__, and that off-by-one is exactly what makes the
    # FIRST while-loop below collect n_rollout+1 raw appends instead of
    # n_rollout, so agent.update()'s own obs_history[-n_rollout:] tail-slice
    # naturally excludes the single garbage placeholder entry
    # (obs_history[0]/walk_actions[0], state -1 / action [None, None], an
    # artifact of prev_observations/prev_actions' pre-construction defaults)
    # that batch_act() always appends on its very first-ever call. Resetting
    # n_walk to 0 here would make the first chunk exactly n_rollout appends
    # (including that placeholder), and agent.update() would then crash in
    # step_to_actions trying to look up [None, None] in poss_actions -
    # confirmed by hitting exactly that crash with n_walk reset to 0, same as
    # how whittington_2020_loop_run.py / tem_training_loop also never reset
    # n_walk before their own first episode.

    loop_idx = 0
    for episode in range(1, N_EPISODES_TOTAL + 1):
        while agent.n_walk < n_rollout:
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

        # Real training step: forward pass, loss, backward, optimizer step,
        # Hebbian memory write, TD update - everything agent.update() already
        # does for normal (non-loop) training, applied here to the ring walk.
        #
        # Episode 1 only: agent.update() excludes the loss term for any state
        # not already in self.visited[env] (a fresh, never-persisted, per-env
        # bookkeeping array - NOT part of the loaded checkpoint's state_dict,
        # so it starts all-False here same as a brand-new agent). The ring
        # loop walks forward through all 20 states without ever repeating
        # one within a single lap, so episode 1's entire first lap is new-
        # state-only -> zero loss terms -> loss.backward() raises (no
        # grad_fn). The forward pass's side effects (Hebbian write, TD
        # update, and marking those 20 states visited) already happened
        # before that point in update(), so this is safe to swallow: episode
        # 2's lap revisits the same 20 states (now marked visited) and gets
        # a real loss normally, for the rest of this run.
        try:
            agent.update()
        except RuntimeError as e:
            if episode == 1 and "does not require grad" in str(e):
                print("  episode 1: first lap visits only new states (no loss term yet) - "
                      "skipping this one gradient step, continuing", flush=True)
            else:
                raise

        if episode == SHIFT_AT_EPISODE:
            print(f"  >>> saving pre_shift_checkpoint at episode {episode}, then applying shift <<<", flush=True)
            save_checkpoint(
                agent, os.path.join(SAVE_ROOT, "pre_shift_checkpoint"), src_model_file, training_dict,
            )
            with open(os.path.join(SAVE_ROOT, "pre_shift_checkpoint", "episode.txt"), "w") as f:
                f.write(str(episode))
            apply_shift(env, agent, cfg)
            new_value_positions = set(cfg["shifted_landmark_states"]) | {cfg["new_reward_xy"]}
            control_states = [xy for xy in control_states if xy not in new_value_positions]

        if ROTATION_ENABLED and episode % CONTROL_SHUFFLE_INTERVAL == 0:
            pool = cfg["non_value_states_post"] if episode >= SHIFT_AT_EPISODE else cfg["non_value_states"]
            control_states = reshuffle_ring_controls(env, cfg, control_states, pool, random)

        if episode % 100 == 0 or episode == 1:
            half = "post-shift" if episode >= SHIFT_AT_EPISODE else "pre-shift"
            print(f"  episode {episode}/{N_EPISODES_TOTAL} [{half}]", flush=True)

    print("Saving final checkpoint...", flush=True)
    save_checkpoint(agent, SAVE_ROOT, src_model_file, training_dict)
    with open(os.path.join(SAVE_ROOT, "state.log"), "w") as f:
        f.write("finished")
    print(f"Done. Results in: {SAVE_ROOT}")


if __name__ == "__main__":
    main()
