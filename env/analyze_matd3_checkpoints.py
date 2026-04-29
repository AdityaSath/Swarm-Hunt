"""Summarize behavior drift across saved MATD3 checkpoints.

Run from ``env/``:
    python analyze_matd3_checkpoints.py --episodes 2
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from typing import Any

import numpy as np
import torch
from agilerl.algorithms.matd3 import MATD3

from swarm_env.config import DRONE_SPEED
from swarm_env.environment import Environment
from swarm_env.policy_actions import action_to_velocity


STAGE_EASY = {
    "prey_speed_factor": 0.35,
    "enable_obstacles": False,
    "always_visible": True,
    "capture_hold_seconds": 1.0,
}
STAGE_FULL = {
    "prey_speed_factor": 1.0,
    "enable_obstacles": True,
    "always_visible": False,
    "capture_hold_seconds": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MATD3 checkpoint behavior")
    parser.add_argument("--checkpoint-dir", default="models/MATD3")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def checkpoint_sort_key(path: str) -> tuple[int, int, str]:
    name = os.path.basename(path)
    if "best" in name:
        return (-1, -1, name)
    match = re.search(r"_gen(\d+)_", name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, 10**9, name)


def checkpoint_label(path: str) -> str:
    name = os.path.basename(path)
    if "best" in name:
        return "best"
    match = re.search(r"_gen(\d+)_", name)
    if match:
        return match.group(1)
    if "final" in name:
        return "final"
    return name


def rollout(
    agent: MATD3,
    env_kwargs: dict[str, Any],
    episodes: int,
    base_seed: int,
    action_repeat: int,
) -> dict[str, float]:
    agent_ids = list(agent.agent_ids)
    idx_to_agent = {i: agent_id for i, agent_id in enumerate(agent_ids)}
    action_space = agent.action_spaces[0]

    captures = 0
    max_holds: list[int] = []
    max_combined: list[int] = []
    speeds: list[float] = []
    alignments: list[float] = []
    mean_distance_deltas: list[float] = []
    threatened_steps = 0
    total_steps = 0
    obstacle_hits = 0
    predator_hits = 0

    for episode in range(episodes):
        env = Environment(
            seed=base_seed + episode,
            drone_count=len(agent_ids),
            **env_kwargs,
        )
        repeat_left = 0
        cached_actions: dict[int, tuple[float, float]] | None = None
        previous_mean_distance = float(np.mean(env._pred_prey_distances()))
        episode_max_hold = 0
        episode_max_combined = 0

        while True:
            if repeat_left <= 0:
                obs_int = env._compute_observations()
                obs = {idx_to_agent[i]: value for i, value in obs_int.items()}
                actions, _ = agent.get_action(obs)
                cached_actions = {}

                for i, agent_id in enumerate(agent_ids):
                    vx, vy = action_to_velocity(
                        np.asarray(actions[agent_id]).reshape(-1),
                        action_space,
                    )
                    cached_actions[i] = (vx, vy)
                    speed = float(math.hypot(vx, vy))
                    speeds.append(speed)

                    if obs_int[i][4] > 0.5:
                        rel = np.array([obs_int[i][5], obs_int[i][6]], dtype=np.float32)
                        rel_norm = float(np.linalg.norm(rel))
                        if speed > 1e-8 and rel_norm > 1e-8:
                            alignments.append(
                                float(np.dot(np.array([vx, vy]), rel) / (speed * rel_norm))
                            )

                repeat_left = max(1, action_repeat)

            assert cached_actions is not None
            _, _, terminations, truncations, infos = env.step(cached_actions)
            repeat_left -= 1
            total_steps += 1
            obstacle_hits += sum(bool(hit) for hit in env._obs_collisions)
            predator_hits += sum(bool(hit) for hit in env._pred_collisions)

            current_mean_distance = float(np.mean(env._pred_prey_distances()))
            mean_distance_deltas.append(previous_mean_distance - current_mean_distance)
            previous_mean_distance = current_mean_distance

            capture = infos["capture"]
            episode_max_hold = max(episode_max_hold, capture.hold_counter)
            episode_max_combined = max(
                episode_max_combined,
                capture.wall_count + capture.in_range_count,
            )
            threatened_steps += int(infos["tactical_state"].name == "THREATENED")

            if any(terminations.values()) or any(truncations.values()):
                captures += int(any(terminations.values()))
                max_holds.append(episode_max_hold)
                max_combined.append(episode_max_combined)
                break

    return {
        "capture_rate": captures / max(1, episodes),
        "max_hold_mean": float(np.mean(max_holds)) if max_holds else 0.0,
        "max_combined_mean": float(np.mean(max_combined)) if max_combined else 0.0,
        "speed_mean": float(np.mean(speeds)) if speeds else 0.0,
        "alignment_mean": float(np.mean(alignments)) if alignments else 0.0,
        "distance_delta_mean": (
            float(np.mean(mean_distance_deltas)) if mean_distance_deltas else 0.0
        ),
        "threatened_frac": threatened_steps / max(1, total_steps),
        "obstacle_hits_per_step": obstacle_hits / max(1, total_steps),
        "predator_hits_per_step": predator_hits / max(1, total_steps),
    }


def main() -> None:
    args = parse_args()
    checkpoint_paths = sorted(
        glob.glob(os.path.join(args.checkpoint_dir, "*.pt")),
        key=checkpoint_sort_key,
    )
    if not checkpoint_paths:
        raise SystemExit(f"No .pt files found in {args.checkpoint_dir}")

    print(
        "checkpoint,easy_capture,easy_hold,easy_combined,easy_speed,"
        "easy_alignment,easy_distance_delta,full_capture,full_hold,"
        "full_combined,full_speed,full_alignment,full_distance_delta,"
        "full_obstacle_hits,full_predator_hits"
    )
    for path in checkpoint_paths:
        agent = MATD3.load(path, device=torch.device("cpu"))
        agent.training = False
        easy = rollout(
            agent,
            STAGE_EASY,
            episodes=args.episodes,
            base_seed=args.seed,
            action_repeat=args.action_repeat,
        )
        full = rollout(
            agent,
            STAGE_FULL,
            episodes=args.episodes,
            base_seed=args.seed + 10_000,
            action_repeat=args.action_repeat,
        )
        print(
            f"{checkpoint_label(path)},"
            f"{easy['capture_rate']:.3f},{easy['max_hold_mean']:.1f},"
            f"{easy['max_combined_mean']:.1f},{easy['speed_mean']:.1f},"
            f"{easy['alignment_mean']:.3f},{easy['distance_delta_mean']:.4f},"
            f"{full['capture_rate']:.3f},{full['max_hold_mean']:.1f},"
            f"{full['max_combined_mean']:.1f},{full['speed_mean']:.1f},"
            f"{full['alignment_mean']:.3f},{full['distance_delta_mean']:.4f},"
            f"{full['obstacle_hits_per_step']:.4f},"
            f"{full['predator_hits_per_step']:.4f}"
        )


if __name__ == "__main__":
    main()
