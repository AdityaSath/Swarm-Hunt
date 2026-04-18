"""Helpers for mapping policy outputs to executed drone velocities."""

from __future__ import annotations

import math
from typing import Any

import gymnasium
import numpy as np

from swarm_env.config import DRONE_SPEED

POLICY_ACTION_LOW = np.array([-1.0, 0.0], dtype=np.float32)
POLICY_ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float32)


def make_policy_action_space() -> gymnasium.spaces.Box:
    """Return the training action space used by MATD3 checkpoints."""
    return gymnasium.spaces.Box(
        low=POLICY_ACTION_LOW,
        high=POLICY_ACTION_HIGH,
        shape=(2,),
        dtype=np.float32,
    )


def uses_policy_action_mapping(action_space: Any) -> bool:
    """Detect whether an action space uses the heading/throttle policy mapping."""
    if action_space is None:
        return False
    low = np.asarray(getattr(action_space, "low", []), dtype=np.float32).reshape(-1)
    high = np.asarray(getattr(action_space, "high", []), dtype=np.float32).reshape(-1)
    return (
        low.shape == (2,)
        and high.shape == (2,)
        and np.allclose(low, POLICY_ACTION_LOW)
        and np.allclose(high, POLICY_ACTION_HIGH)
    )


def policy_action_to_velocity(action: np.ndarray | list[float] | tuple[float, float]) -> tuple[float, float]:
    """Map a policy action `(heading, throttle)` to world velocity `(vx, vy)`."""
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 2:
        raise ValueError(f"Expected 2-D action, got shape {arr.shape}")

    heading = float(np.clip(arr[0], POLICY_ACTION_LOW[0], POLICY_ACTION_HIGH[0]))
    throttle = float(np.clip(arr[1], POLICY_ACTION_LOW[1], POLICY_ACTION_HIGH[1]))
    angle = heading * math.pi
    speed = throttle * DRONE_SPEED
    return speed * math.cos(angle), speed * math.sin(angle)


def action_to_velocity(
    action: np.ndarray | list[float] | tuple[float, float],
    action_space: Any,
) -> tuple[float, float]:
    """Decode either a new heading/throttle action or a legacy velocity action."""
    if uses_policy_action_mapping(action_space):
        return policy_action_to_velocity(action)

    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 2:
        raise ValueError(f"Expected 2-D action, got shape {arr.shape}")
    return float(arr[0]), float(arr[1])
