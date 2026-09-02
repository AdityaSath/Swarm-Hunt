"""Tests for acceleration-limited predator movement."""

import pytest

from swarm_env.config import DRONE_MAX_ACCELERATION, DRONE_SPEED, DT
from swarm_env.drone import Drone


def test_acceleration_is_limited_from_rest():
    drone = Drone(100.0, 100.0)
    drone.set_desired_velocity(DRONE_SPEED, 0.0)

    drone.integrate(DT)

    expected_speed = DRONE_MAX_ACCELERATION * DT
    assert drone.velocity.x == pytest.approx(expected_speed)
    assert drone.velocity.y == pytest.approx(0.0)
    assert drone.position.x == pytest.approx(100.0 + expected_speed * DT)


def test_velocity_reaches_but_never_exceeds_command():
    drone = Drone(100.0, 100.0)
    commanded_speed = DRONE_SPEED * 0.5
    drone.set_desired_velocity(commanded_speed, 0.0)

    for _ in range(60):
        drone.integrate(DT)

    assert drone.velocity.x == pytest.approx(commanded_speed)
    assert drone.velocity.y == pytest.approx(0.0)


def test_desired_velocity_is_speed_clipped():
    drone = Drone(100.0, 100.0)
    drone.set_desired_velocity(DRONE_SPEED * 10.0, 0.0)

    assert drone.desired_velocity.length() == pytest.approx(DRONE_SPEED)


def test_reversal_is_acceleration_limited():
    drone = Drone(100.0, 100.0, vx=DRONE_SPEED)
    drone.set_desired_velocity(-DRONE_SPEED, 0.0)
    before = drone.velocity.copy()

    drone.integrate(DT)

    assert (drone.velocity - before).length() == pytest.approx(
        DRONE_MAX_ACCELERATION * DT
    )
    assert drone.velocity.x > -DRONE_SPEED
