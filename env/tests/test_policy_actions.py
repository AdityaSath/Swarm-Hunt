"""Tests for the MATD3 policy-action decoding path."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swarm_env.config import DRONE_SPEED
from swarm_env.policy_actions import (
    POLICY_ACTION_HIGH,
    POLICY_ACTION_LOW,
    action_to_velocity,
    make_policy_action_space,
    policy_action_to_velocity,
    uses_policy_action_mapping,
)


def test_policy_action_space_bounds():
    space = make_policy_action_space()
    assert np.allclose(space.low, POLICY_ACTION_LOW)
    assert np.allclose(space.high, POLICY_ACTION_HIGH)
    assert uses_policy_action_mapping(space)


def test_policy_action_to_velocity_mapping():
    vx, vy = policy_action_to_velocity((0.0, 1.0))
    assert math.isclose(vx, DRONE_SPEED, rel_tol=1e-6)
    assert math.isclose(vy, 0.0, abs_tol=1e-6)

    vx, vy = policy_action_to_velocity((0.5, 1.0))
    assert math.isclose(vx, 0.0, abs_tol=1e-6)
    assert math.isclose(vy, DRONE_SPEED, rel_tol=1e-6)

    vx, vy = policy_action_to_velocity((0.0, 0.0))
    assert math.isclose(vx, 0.0, abs_tol=1e-6)
    assert math.isclose(vy, 0.0, abs_tol=1e-6)


def test_legacy_velocity_actions_passthrough():
    legacy_space = type(
        "LegacySpace",
        (),
        {
            "low": np.array([-DRONE_SPEED, -DRONE_SPEED], dtype=np.float32),
            "high": np.array([DRONE_SPEED, DRONE_SPEED], dtype=np.float32),
        },
    )()
    vx, vy = action_to_velocity((12.5, -7.0), legacy_space)
    assert math.isclose(vx, 12.5, rel_tol=1e-6)
    assert math.isclose(vy, -7.0, rel_tol=1e-6)


if __name__ == "__main__":
    test_policy_action_space_bounds()
    test_policy_action_to_velocity_mapping()
    test_legacy_velocity_actions_passthrough()
    print("PASS: policy action decoding")
