"""Targeted tests for per-drone reward assignment."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swarm_env.capture import CaptureStatus, PreyTacticalState
from swarm_env.config import (
    DIST_SHAPING_CLIP,
    DIST_SHAPING_SCALE,
    DRONE_SPEED,
    IDLE_SPEED_THRESHOLD,
    MAX_STEPS,
    PENALTY_IDLE,
    PENALTY_OBSTACLE_COLLISION,
    PENALTY_PREDATOR_COLLISION,
    PENALTY_VISIBLE_MISALIGNMENT,
    REWARD_COMBINED_PROGRESS,
    REWARD_CAPTURE_CONTRIBUTOR,
    REWARD_CAPTURE_RANGE_STEP,
    REWARD_CAPTURE_TEAM,
    REWARD_HOLD_PROGRESS,
    REWARD_THREATENED,
    REWARD_TIMEOUT,
    REWARD_VISIBLE_ALIGNMENT,
    R_CAPTURE_RANGE,
    WORLD_SCALE,
)
from swarm_env.environment import Environment


def _set_non_idle_velocities(env: Environment) -> None:
    for drone in env.drones:
        drone.velocity.update(IDLE_SPEED_THRESHOLD + 1.0, 0.0)


def _disable_prey_shaping(env: Environment) -> None:
    env.prey = None
    env._prev_predator_distances = [0.0 for _ in env.drones]


def test_capture_reward_is_shared_with_contributor_bonus():
    env = Environment(seed=0)
    _set_non_idle_velocities(env)
    _disable_prey_shaping(env)
    env._prev_capture_combined = 2
    rewards = env._compute_rewards(
        CaptureStatus(in_range_count=2, contributor_indices=[1, 3], hold_counter=0, wall_count=0),
        PreyTacticalState.CAPTURED,
    )

    assert rewards[1] == REWARD_CAPTURE_TEAM + REWARD_CAPTURE_CONTRIBUTOR
    assert rewards[3] == REWARD_CAPTURE_TEAM + REWARD_CAPTURE_CONTRIBUTOR

    for idx, reward in rewards.items():
        if idx not in (1, 3):
            assert reward == REWARD_CAPTURE_TEAM


def test_timeout_and_collision_penalties_are_per_drone():
    env = Environment(seed=0)
    _set_non_idle_velocities(env)
    _disable_prey_shaping(env)
    env._step_count = MAX_STEPS
    env._obs_collisions = [False] * len(env.drones)
    env._pred_collisions = [False] * len(env.drones)
    env._obs_collisions[0] = True
    env._pred_collisions[1] = True

    rewards = env._compute_rewards(
        CaptureStatus(in_range_count=0, contributor_indices=[], hold_counter=0, wall_count=0),
        PreyTacticalState.FREE,
    )

    assert rewards[0] == REWARD_TIMEOUT + PENALTY_OBSTACLE_COLLISION
    assert rewards[1] == REWARD_TIMEOUT + PENALTY_PREDATOR_COLLISION

    for idx, reward in rewards.items():
        if idx not in (0, 1):
            assert reward == REWARD_TIMEOUT


def test_progress_and_threatened_rewards_are_shared():
    env = Environment(seed=0)
    _set_non_idle_velocities(env)
    _disable_prey_shaping(env)
    env._prev_tactical = PreyTacticalState.FREE
    env._prev_capture_combined = 1
    env._prev_hold_counter = 0

    rewards = env._compute_rewards(
        CaptureStatus(in_range_count=2, contributor_indices=[0, 1], hold_counter=1, wall_count=1),
        PreyTacticalState.THREATENED,
    )

    expected = REWARD_THREATENED + (2 * REWARD_COMBINED_PROGRESS) + REWARD_HOLD_PROGRESS
    for reward in rewards.values():
        assert reward >= expected


def test_distance_shaping_rewards_individual_progress():
    env = Environment(seed=0)
    _set_non_idle_velocities(env)
    env.prey = None
    env._prev_predator_distances = [10.0 for _ in env.drones]
    env._pred_prey_distances = lambda: [9.0] + [10.0 for _ in env.drones[1:]]

    rewards = env._compute_rewards(
        CaptureStatus(in_range_count=0, contributor_indices=[], hold_counter=0, wall_count=0),
        PreyTacticalState.FREE,
    )

    assert rewards[0] == min(
        DIST_SHAPING_CLIP,
        (1.0 / WORLD_SCALE) * DIST_SHAPING_SCALE,
    )
    for idx in range(1, len(env.drones)):
        assert rewards[idx] == 0.0


def test_idle_penalty_is_per_drone():
    env = Environment(seed=0)
    _set_non_idle_velocities(env)
    _disable_prey_shaping(env)
    env.drones[0].velocity.update(0.0, 0.0)

    rewards = env._compute_rewards(
        CaptureStatus(in_range_count=0, contributor_indices=[], hold_counter=0, wall_count=0),
        PreyTacticalState.FREE,
    )

    assert rewards[0] == PENALTY_IDLE
    for idx in range(1, len(env.drones)):
        assert rewards[idx] == 0.0


def test_visible_alignment_rewards_pursuit_direction():
    env = Environment(
        seed=0,
        drone_count=1,
        enable_obstacles=False,
        always_visible=True,
        prey_speed_factor=0.0,
    )
    assert env.prey is not None
    env.drones[0].position.update(100.0, 200.0)
    env.drones[0].velocity.update(DRONE_SPEED, 0.0)
    env.prey.position.update(300.0, 200.0)
    env.prey.velocity.update(0.0, 0.0)
    env._prev_predator_distances = env._pred_prey_distances()

    rewards = env._compute_rewards(
        CaptureStatus(in_range_count=0, contributor_indices=[], hold_counter=0, wall_count=0),
        PreyTacticalState.FREE,
    )

    assert rewards[0] == REWARD_VISIBLE_ALIGNMENT

    env.drones[0].velocity.update(-DRONE_SPEED, 0.0)
    env._prev_predator_distances = env._pred_prey_distances()
    rewards = env._compute_rewards(
        CaptureStatus(in_range_count=0, contributor_indices=[], hold_counter=0, wall_count=0),
        PreyTacticalState.FREE,
    )

    assert rewards[0] == PENALTY_VISIBLE_MISALIGNMENT


def test_capture_range_step_reward_is_per_contributor():
    env = Environment(
        seed=0,
        drone_count=1,
        enable_obstacles=False,
        always_visible=True,
        prey_speed_factor=0.0,
    )
    assert env.prey is not None
    env.prey.position.update(300.0, 200.0)
    env.prey.velocity.update(0.0, 0.0)
    env.drones[0].position.update(300.0 + R_CAPTURE_RANGE * 0.5, 200.0)
    env.drones[0].velocity.update(0.0, DRONE_SPEED)
    env._prev_predator_distances = env._pred_prey_distances()

    rewards = env._compute_rewards(
        CaptureStatus(in_range_count=1, contributor_indices=[0], hold_counter=0, wall_count=0),
        PreyTacticalState.THREATENED,
    )

    assert rewards[0] >= REWARD_CAPTURE_RANGE_STEP


if __name__ == "__main__":
    test_capture_reward_is_shared_with_contributor_bonus()
    test_timeout_and_collision_penalties_are_per_drone()
    test_progress_and_threatened_rewards_are_shared()
    test_distance_shaping_rewards_individual_progress()
    test_idle_penalty_is_per_drone()
    test_visible_alignment_rewards_pursuit_direction()
    test_capture_range_step_reward_is_per_contributor()
    print("PASS: per-drone reward assignment")
