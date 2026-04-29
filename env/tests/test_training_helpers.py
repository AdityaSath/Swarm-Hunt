"""Tests for MATD3 training helper behavior."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swarm_env.config import DRONE_SPEED
from swarm_env.policy_actions import (
    MIN_POLICY_THROTTLE,
    action_to_velocity,
    make_policy_action_space,
)
from train import _eval_metrics_better, sample_vectorized_random_actions


def test_vectorized_random_actions_decode_to_motion():
    agent_ids = ["predator_0", "predator_1"]
    action_space = make_policy_action_space()
    actions = sample_vectorized_random_actions(
        agent_ids,
        {agent_id: action_space for agent_id in agent_ids},
        num_envs=4,
    )

    assert set(actions) == set(agent_ids)
    for agent_id in agent_ids:
        assert actions[agent_id].shape == (4, 2)
        for action in actions[agent_id]:
            vx, vy = action_to_velocity(action, action_space)
            speed = math.hypot(vx, vy)
            assert speed >= DRONE_SPEED * MIN_POLICY_THROTTLE


def test_eval_tie_breaker_prefers_moving_policy():
    stationary = {
        "capture_rate": 0.0,
        "max_hold_mean": 0.0,
        "max_combined_mean": 1.0,
        "visible_alignment_mean": 0.0,
        "action_speed_mean": 0.0,
    }
    moving = dict(stationary)
    moving["action_speed_mean"] = DRONE_SPEED * MIN_POLICY_THROTTLE

    assert _eval_metrics_better(moving, stationary)
    assert not _eval_metrics_better(stationary, moving)


if __name__ == "__main__":
    test_vectorized_random_actions_decode_to_motion()
    test_eval_tie_breaker_prefers_moving_policy()
    print("PASS: MATD3 training helpers")
