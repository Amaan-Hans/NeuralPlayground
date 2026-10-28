"""Diagram of the ring-loop object layout: a 10x10 room grid with the
20-state ring highlighted, showing exactly which states are value
landmarks / the reward / control landmarks / filler, before and after the
episode-250 shift. Saved into Thesis/quantitative/loop/.

Controls are drawn as ONE representative random sample from the 15-state
non-value pool (reproducible via a fixed seed) - for the 4 rotation-ON
conditions this set actually changes every 25 episodes during the real
probe; for the 4 "noshuffle" conditions it's fixed for the whole run, same
as drawn here.
"""
import os
import random
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

AGENT_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AGENT_EXAMPLES_DIR)
from probe_loop_ring import (  # noqa: E402
    RING_LOOP, LOOP_START, _simulate_path, REWARD_RING_IDX, N_RING_LANDMARKS, N_RING_CONTROLS, ROOM_SIZE,
    value_block,
)

OUT_DIR = r"C:\Users\hansl\Desktop\Projects\Masters_research\Thesis\quantitative\loop"
LAYOUT_SEED = 42

BLUE = "#2a78d6"     # value landmarks
ORANGE = "#eb6834"   # control landmarks
GOLD = "#eda100"     # reward
FILLER = "#e1e0d9"   # filler / null states
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

ROOM_N = int(ROOM_SIZE)  # 10


def ring_geometry():
    full_path = [LOOP_START] + _simulate_path(LOOP_START, RING_LOOP, step_size=1)
    unique_path_states = full_path[:-1]
    n_ring = len(unique_path_states)

    landmark_idx, reward_idx, open_idx = value_block(REWARD_RING_IDX, n_ring)
    landmark_states = [unique_path_states[i] for i in landmark_idx]
    reward_xy = unique_path_states[reward_idx]
    value_indices = set(landmark_idx) | {reward_idx, open_idx}
    non_value_states = [unique_path_states[i] for i in range(n_ring) if i not in value_indices]

    # Post-shift block's own reserved indices (landmarks+reward move forward
    # by 1; the open slot moves forward by 1 too, same relative structure).
    new_landmark_idx, new_reward_idx, new_open_idx = value_block(REWARD_RING_IDX + 1, n_ring)
    shifted_landmark_states = [unique_path_states[i] for i in new_landmark_idx]
    new_reward_xy = unique_path_states[new_reward_idx]
    new_value_indices = set(new_landmark_idx) | {new_reward_idx, new_open_idx}
    non_value_states_post = [unique_path_states[i] for i in range(n_ring) if i not in new_value_indices]
    open_xy_post = unique_path_states[new_open_idx]

    random.seed(LAYOUT_SEED)
    control_states_pre = random.sample(non_value_states, N_RING_CONTROLS)
    # Represent a fresh reshuffle (rotation-ON conditions only) for the post
    # panel - drawn from the POST-shift block's own non-value pool, not the
    # pre-shift one (which still includes the old landmark/reward/open
    # positions, now wrong once the block has moved).
    control_states_post = random.sample(non_value_states_post, N_RING_CONTROLS)

    return {
        "landmark_pre": landmark_states, "reward_pre": reward_xy, "control_pre": control_states_pre,
        "open_pre": unique_path_states[open_idx],
        "landmark_post": shifted_landmark_states, "reward_post": new_reward_xy, "control_post": control_states_post,
        "open_post": open_xy_post,
    }


def xy_to_cell(xy):
    """Map a state's (x, y) coordinate to a 0-indexed (row, col) grid cell."""
    col = int(round(xy[0] + ROOM_N / 2 - 0.5))
    row = int(round(xy[1] + ROOM_N / 2 - 0.5))
    return row, col


def draw_panel(ax, title, subtitle, landmarks, reward, controls, open_slot):
    ax.set_facecolor(SURFACE)
    for row in range(ROOM_N):
        for col in range(ROOM_N):
            ax.add_patch(Rectangle((col, row), 1, 1, facecolor=FILLER, edgecolor="white", linewidth=1.2))

    for i, xy in enumerate(landmarks):
        row, col = xy_to_cell(xy)
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=BLUE, edgecolor="white", linewidth=1.2))
        ax.text(col + 0.5, row + 0.5, str(i), ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    row, col = xy_to_cell(reward)
    ax.add_patch(Rectangle((col, row), 1, 1, facecolor=GOLD, edgecolor="white", linewidth=1.2))
    ax.text(col + 0.5, row + 0.5, "R", ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    for i, xy in enumerate(controls):
        row, col = xy_to_cell(xy)
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=ORANGE, edgecolor="white", linewidth=1.2))
        ax.text(col + 0.5, row + 0.5, str(10 + i), ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    row, col = xy_to_cell(open_slot)
    ax.add_patch(Rectangle((col, row), 1, 1, facecolor=FILLER, edgecolor=GOLD, linewidth=2.5, linestyle="--"))

    ax.set_xlim(0, ROOM_N)
    ax.set_ylim(0, ROOM_N)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, fontweight="bold", pad=10)
    ax.text(0.5, -0.04, subtitle, transform=ax.transAxes, ha="center", va="top",
            color=INK_MUTED, fontsize=8)


def main():
    geo = ring_geometry()

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2), facecolor=PAGE)
    draw_panel(axes[0], "Pre-shift (episodes 0-249)",
               "controls shown: one random draw from the pool",
               geo["landmark_pre"], geo["reward_pre"], geo["control_pre"], geo["open_pre"])
    draw_panel(axes[1], "Post-shift (episodes 250-499)",
               "controls shown: one possible post-reshuffle draw (rotation-ON only)",
               geo["landmark_post"], geo["reward_post"], geo["control_post"], geo["open_post"])

    legend_items = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, label="value landmark (id 0-4)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=GOLD, label="reward (R)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, label="control landmark (id 10-14)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=FILLER, label="filler / null"),
        plt.Rectangle((0, 0), 1, 1, facecolor=FILLER, edgecolor=GOLD, linewidth=2, linestyle="--",
                       label="reserved open slot (shift target)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5, frameon=False, fontsize=9,
               labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("Ring-loop object layout: 10x10 room, 20-state ring (inner 6x6 border)",
                 color=INK_PRIMARY, fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.90,
              "5 value landmarks sit at fixed, spaced-out distances before reward (1, 3, 5, 7, 9 ring-states away -\n"
              "not contiguous); the ring-state right after reward is always kept open. The remaining ring states\n"
              "are a pool where 5 hold controls and the rest stay filler. At episode 250 the whole landmark+reward\n"
              "block shifts FORWARD by one ring-state into that open slot. For the 4 rotation-ON conditions, which\n"
              "states in the pool hold controls is re-randomized every 25 episodes; for the 4 \"noshuffle\"\n"
              "conditions that draw is fixed for the whole run.",
              ha="center", va="top", color=INK_MUTED, fontsize=8.5)
    fig.tight_layout(rect=[0, 0.07, 1, 0.78])

    out_path = os.path.join(OUT_DIR, "loop_layout.png")
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
