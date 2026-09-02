"""Integration tests for the scripted formation showcase controller."""

import pytest

from swarm_env.config import DRONE_SPEED, MAX_STEPS
from swarm_env.environment import Environment
from swarm_env.formation_controller import FormationController


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_controller_captures_with_default_obstacles(seed):
    env = Environment(seed=seed, prey_speed_factor=0.5)
    controller = FormationController()
    controller.reset(env)

    for _ in range(MAX_STEPS):
        actions = controller.actions(env)
        assert all(
            vx * vx + vy * vy <= DRONE_SPEED * DRONE_SPEED + 1e-6
            for vx, vy in actions.values()
        )
        _, _, terminations, truncations, _ = env.step(actions)
        if any(terminations.values()):
            return
        if any(truncations.values()):
            break

    pytest.fail("formation controller did not capture before timeout")
