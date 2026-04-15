"""Targeted tests for per-drone reward assignment."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swarm_env.capture import PreyTacticalState
from swarm_env.config import (
    MAX_STEPS,
    PENALTY_OBSTACLE_COLLISION,
    PENALTY_PREDATOR_COLLISION,
    REWARD_CAPTURE,
    REWARD_TIMEOUT,
)
from swarm_env.environment import Environment


def test_capture_reward_only_goes_to_contributors():
    env = Environment(seed=0)
    rewards = env._compute_rewards([1, 3], PreyTacticalState.CAPTURED)

    assert rewards[1] == REWARD_CAPTURE
    assert rewards[3] == REWARD_CAPTURE

    for idx, reward in rewards.items():
        if idx not in (1, 3):
            assert reward == 0.0


def test_timeout_and_collision_penalties_are_per_drone():
    env = Environment(seed=0)
    env._step_count = MAX_STEPS
    env._obs_collisions = [False] * len(env.drones)
    env._pred_collisions = [False] * len(env.drones)
    env._obs_collisions[0] = True
    env._pred_collisions[1] = True

    rewards = env._compute_rewards([], PreyTacticalState.FREE)

    assert rewards[0] == REWARD_TIMEOUT + PENALTY_OBSTACLE_COLLISION
    assert rewards[1] == REWARD_TIMEOUT + PENALTY_PREDATOR_COLLISION

    for idx, reward in rewards.items():
        if idx not in (0, 1):
            assert reward == REWARD_TIMEOUT


if __name__ == "__main__":
    test_capture_reward_only_goes_to_contributors()
    test_timeout_and_collision_penalties_are_per_drone()
    print("PASS: per-drone reward assignment")