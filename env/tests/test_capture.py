"""
Tests for distance-based combo capture FSM and prey hiding.

Scenario A: 4 predators inside R_CAPTURE_RANGE, open arena → CAPTURED after hold
Scenario B: only 3 predators, no walls → never CAPTURED
Scenario C: 3 predators + 1 wall intersecting blue circle → CAPTURED after hold
Scenario D: prey hiding longer than T_HIDE_MAX → forced exit
"""

import math
import os
import sys

import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swarm_env.config import (
    ARENA_WIDTH,
    ARENA_HEIGHT,
    R_CAPTURE_RANGE,
    CAPTURE_HOLD_STEPS,
    COMBO_CAPTURE_NEED,
    T_HIDE_MAX,
    PREY_RADIUS,
)
from swarm_env.capture import TacticalFSM, PreyTacticalState, walls_intersecting_capture_circle
from swarm_env.prey import Prey


def _make_positions_around(
    cx: float, cy: float, n: int, radius: float
) -> list[tuple[float, float]]:
    return [
        (
            cx + radius * math.cos(2 * math.pi * i / n),
            cy + radius * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]


def _fsm_step(
    fsm: TacticalFSM,
    preds: list[tuple[float, float]],
    px: float,
    py: float,
) -> PreyTacticalState:
    return fsm.update(preds, px, py, float(ARENA_WIDTH), float(ARENA_HEIGHT))


def test_capture_4_drones_open_arena():
    """Center prey, four drones in ring → walls=0, drones=4 → combo ≥ need."""
    prey_x = ARENA_WIDTH / 2
    prey_y = ARENA_HEIGHT / 2
    dist = R_CAPTURE_RANGE * 0.5
    preds = _make_positions_around(prey_x, prey_y, COMBO_CAPTURE_NEED, dist)
    assert walls_intersecting_capture_circle(prey_x, prey_y, ARENA_WIDTH, ARENA_HEIGHT) == 0

    fsm = TacticalFSM()
    for step in range(CAPTURE_HOLD_STEPS - 1):
        state = _fsm_step(fsm, preds, prey_x, prey_y)
        assert state != PreyTacticalState.CAPTURED, f"early capture at {step}"

    state = _fsm_step(fsm, preds, prey_x, prey_y)
    assert state == PreyTacticalState.CAPTURED


def test_no_capture_3_drones_open_arena():
    prey_x = ARENA_WIDTH / 2
    prey_y = ARENA_HEIGHT / 2
    dist = R_CAPTURE_RANGE * 0.5
    preds = _make_positions_around(prey_x, prey_y, 3, dist)

    fsm = TacticalFSM()
    for _ in range(CAPTURE_HOLD_STEPS * 3):
        state = _fsm_step(fsm, preds, prey_x, prey_y)
        assert state != PreyTacticalState.CAPTURED


def test_capture_3_drones_plus_wall():
    """Prey near left wall so one wall intersects blue circle; 3 drones in ring → 1+3=4."""
    prey_x = R_CAPTURE_RANGE * 0.45
    prey_y = ARENA_HEIGHT / 2
    assert walls_intersecting_capture_circle(prey_x, prey_y, ARENA_WIDTH, ARENA_HEIGHT) >= 1

    dist = R_CAPTURE_RANGE * 0.5
    preds = _make_positions_around(prey_x, prey_y, 3, dist)

    fsm = TacticalFSM()
    for step in range(CAPTURE_HOLD_STEPS - 1):
        state = _fsm_step(fsm, preds, prey_x, prey_y)
        assert state != PreyTacticalState.CAPTURED, f"early at {step}"
    state = _fsm_step(fsm, preds, prey_x, prey_y)
    assert state == PreyTacticalState.CAPTURED


def test_capture_with_custom_hold_steps():
    prey_x = ARENA_WIDTH / 2
    prey_y = ARENA_HEIGHT / 2
    preds = _make_positions_around(prey_x, prey_y, COMBO_CAPTURE_NEED, R_CAPTURE_RANGE * 0.5)

    fsm = TacticalFSM(capture_hold_steps=3)
    for _ in range(2):
        assert _fsm_step(fsm, preds, prey_x, prey_y) != PreyTacticalState.CAPTURED
    assert _fsm_step(fsm, preds, prey_x, prey_y) == PreyTacticalState.CAPTURED


def test_prey_forced_exit_after_hide_max():
    pygame.init()

    prey = Prey(100, 100, radius=PREY_RADIUS)
    obs_rect = pygame.Rect(50, 50, 100, 100)
    obstacle_rects = [obs_rect]
    pred_positions = [(200, 100)]

    prey.hiding = True
    prey.hide_steps = 0
    prey.velocity = pygame.math.Vector2(0, 0)

    for step in range(T_HIDE_MAX + 5):
        prey.decide(pred_positions, obstacle_rects, ARENA_WIDTH, ARENA_HEIGHT)
        if step < T_HIDE_MAX - 1:
            assert prey.hiding or prey.velocity.length() > 0
        if step == T_HIDE_MAX - 1:
            assert not prey.hiding
            assert prey.velocity.length() > 0
            break

    pygame.quit()


if __name__ == "__main__":
    test_capture_4_drones_open_arena()
    print("PASS: Scenario A — 4 drones, open arena")

    test_no_capture_3_drones_open_arena()
    print("PASS: Scenario B — 3 drones only, no capture")

    test_capture_3_drones_plus_wall()
    print("PASS: Scenario C — 3 drones + wall → capture")

    test_capture_with_custom_hold_steps()
    print("PASS: Scenario C2 — custom hold steps")

    test_prey_forced_exit_after_hide_max()
    print("PASS: Scenario D — T_HIDE_MAX forced exit")

    print("\nAll targeted tests passed.")
