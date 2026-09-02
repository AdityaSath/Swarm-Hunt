"""Contracts shared by expert collection, BC, and MATD3 fine-tuning."""

import math

import numpy as np
import pytest

from swarm_env.capture import PreyTacticalState, TacticalFSM
from swarm_env.config import (
    ARENA_HEIGHT,
    ARENA_WIDTH,
    DRONE_COUNT,
    FORMATION_TARGET_RADIUS,
)
from swarm_env.environment import Environment, OBS_SIZE
from train import combine_done


def test_observations_expose_unique_stable_roles():
    env = Environment(seed=12)
    observations, _ = env.reset(seed=12)

    assert OBS_SIZE == 66
    role_vectors = [tuple(observations[i][4:6]) for i in range(DRONE_COUNT)]
    assert len(set(role_vectors)) == DRONE_COUNT

    for i, (sin_angle, cos_angle) in enumerate(role_vectors):
        expected = env.formation_role_angle(i)
        assert sin_angle == np.float32(math.sin(expected))
        assert cos_angle == np.float32(math.cos(expected))


def test_prey_start_position_changes_across_seeds():
    env = Environment()
    positions = []
    for seed in range(5):
        env.reset(seed=seed)
        assert env.prey is not None
        positions.append((env.prey.position.x, env.prey.position.y))
    assert len(set(positions)) == len(positions)


def test_formation_targets_use_capture_ring_radius_away_from_walls():
    env = Environment(seed=2, obstacles_enabled=False)
    assert env.prey is not None
    env.prey.position.update(ARENA_WIDTH / 2, ARENA_HEIGHT / 2)

    for i in range(env.num_agents):
        assert env.formation_target(i).distance_to(env.prey.position) == pytest.approx(
            FORMATION_TARGET_RADIUS
        )


def test_capture_hold_is_configurable_for_curriculum():
    hold_steps = 3
    fsm = TacticalFSM(capture_hold_steps=hold_steps, combo_capture_need=1)
    predators = [(ARENA_WIDTH / 2 + 20.0, ARENA_HEIGHT / 2)]

    for _ in range(hold_steps - 1):
        state = fsm.update(
            predators,
            ARENA_WIDTH / 2,
            ARENA_HEIGHT / 2,
            ARENA_WIDTH,
            ARENA_HEIGHT,
        )
        assert state != PreyTacticalState.CAPTURED
    assert fsm.update(
        predators,
        ARENA_WIDTH / 2,
        ARENA_HEIGHT / 2,
        ARENA_WIDTH,
        ARENA_HEIGHT,
    ) == PreyTacticalState.CAPTURED


def test_replay_done_combines_termination_and_timeout():
    termination = {"predator_0": np.asarray([True, False, False])}
    truncation = {"predator_0": np.asarray([False, True, False])}
    done = combine_done(termination, truncation)
    assert done["predator_0"].tolist() == [True, True, False]
