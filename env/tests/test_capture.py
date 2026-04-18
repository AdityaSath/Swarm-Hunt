"""
Tests for distance-based combo capture FSM.

Scenario A: 4 predators inside R_CAPTURE_RANGE, open arena → CAPTURED after hold
Scenario B: only 3 predators, no walls → never CAPTURED
Scenario C: 3 predators + 1 wall intersecting blue circle → CAPTURED after hold
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swarm_env.config import (
    ARENA_WIDTH,
    ARENA_HEIGHT,
    R_CAPTURE_RANGE,
    CAPTURE_HOLD_STEPS,
    COMBO_CAPTURE_NEED,
)
from swarm_env.capture import TacticalFSM, PreyTacticalState, walls_intersecting_capture_circle


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


def test_no_capture_below_combo_open_arena():
    """Only COMBO_CAPTURE_NEED-1 predators in the ring, no walls -> never CAPTURED."""
    prey_x = ARENA_WIDTH / 2
    prey_y = ARENA_HEIGHT / 2
    dist = R_CAPTURE_RANGE * 0.5
    preds = _make_positions_around(prey_x, prey_y, COMBO_CAPTURE_NEED - 1, dist)

    fsm = TacticalFSM()
    for _ in range(CAPTURE_HOLD_STEPS * 3):
        state = _fsm_step(fsm, preds, prey_x, prey_y)
        assert state != PreyTacticalState.CAPTURED


def test_capture_with_wall_assist():
    """Prey near left wall so one wall intersects blue circle; (NEED-1) drones in ring."""
    prey_x = R_CAPTURE_RANGE * 0.45
    prey_y = ARENA_HEIGHT / 2
    assert walls_intersecting_capture_circle(prey_x, prey_y, ARENA_WIDTH, ARENA_HEIGHT) >= 1

    dist = R_CAPTURE_RANGE * 0.5
    preds = _make_positions_around(prey_x, prey_y, COMBO_CAPTURE_NEED - 1, dist)

    fsm = TacticalFSM()
    for step in range(CAPTURE_HOLD_STEPS - 1):
        state = _fsm_step(fsm, preds, prey_x, prey_y)
        assert state != PreyTacticalState.CAPTURED, f"early at {step}"
    state = _fsm_step(fsm, preds, prey_x, prey_y)
    assert state == PreyTacticalState.CAPTURED


if __name__ == "__main__":
    test_capture_4_drones_open_arena()
    print("PASS: Scenario A — 4 drones, open arena")

    test_no_capture_below_combo_open_arena()
    print("PASS: Scenario B — sub-combo drones only, no capture")

    test_capture_with_wall_assist()
    print("PASS: Scenario C — drones + wall → capture")

    print("\nAll targeted tests passed.")
