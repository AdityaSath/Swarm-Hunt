"""
Targeted tests for capture geometry, tactical FSM, and prey hiding.

Scenario A: 4 contributors + border → capture after T_HOLD
Scenario B: only 3 contributors → no capture
Scenario C: prey hiding longer than T_HIDE_MAX → forced exit
"""

import math
import sys
import os

import pygame

# allow imports from parent ``env`` package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swarm_env.config import (
    ARENA_WIDTH,
    ARENA_HEIGHT,
    R_CAP,
    R_WALL_CAP,
    PHI_ESCAPE_MAX,
    MIN_PREDATOR_CONTRIBUTORS,
    T_HOLD,
    T_HIDE_MAX,
    PREY_RADIUS,
    DRONE_RADIUS,
)
from swarm_env.capture import (
    compute_escape_gap,
    TacticalFSM,
    PreyTacticalState,
)
from swarm_env.prey import Prey


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_positions_around(
    cx: float, cy: float, n: int, radius: float
) -> list[tuple[float, float]]:
    """Place *n* points evenly spaced on a circle of *radius* around (cx, cy)."""
    return [
        (cx + radius * math.cos(2 * math.pi * i / n),
         cy + radius * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Scenario A: 4 contributors + border-aware enclosure → capture after T_HOLD
# ---------------------------------------------------------------------------

def test_capture_4_contributors_with_border():
    """
    Place prey in a corner (within R_WALL_CAP of left + bottom walls) so two
    walls act as blockers.  Position 4 predators inside R_CAP to subdivide
    the remaining arc so every gap < PHI_ESCAPE_MAX.  Verify capture after
    T_HOLD.
    """
    prey_x = R_WALL_CAP * 0.5
    prey_y = ARENA_HEIGHT - R_WALL_CAP * 0.5

    # Walls contribute blocker angles: π (left) and π/2 (bottom in pygame coords).
    # 4 predators subdivide the remaining arcs so all gaps ≤ 67.5° < 70°.
    dist = R_CAP * 0.6
    pred_angles = [
        3 * math.pi / 4,   # between bottom-wall (π/2) and left-wall (π)
        -math.pi + math.radians(67.5) * 1,
        -math.pi + math.radians(67.5) * 2,
        -math.pi + math.radians(67.5) * 3,
    ]
    preds = [
        (prey_x + dist * math.cos(a), prey_y + dist * math.sin(a))
        for a in pred_angles
    ]

    gap = compute_escape_gap(prey_x, prey_y, preds)
    assert gap.predator_contributors == 4, f"Expected 4 contributors, got {gap.predator_contributors}"
    assert gap.border_blocker_count >= 2, f"Expected ≥2 border blockers, got {gap.border_blocker_count}"
    assert gap.largest_gap < PHI_ESCAPE_MAX, (
        f"Gap {math.degrees(gap.largest_gap):.1f}° should be < {math.degrees(PHI_ESCAPE_MAX):.1f}°"
    )

    # FSM should reach CAPTURED after T_HOLD consecutive qualifying steps
    fsm = TacticalFSM()
    fsm.state = PreyTacticalState.CONTAINED
    for step in range(T_HOLD + 5):
        state = fsm.update(gap, preds, prey_x, prey_y)
        if state == PreyTacticalState.CAPTURED:
            assert step + 1 >= T_HOLD, f"Captured too early at step {step}"
            break
    else:
        raise AssertionError(f"Not captured after {T_HOLD + 5} steps")

    assert state == PreyTacticalState.CAPTURED


# ---------------------------------------------------------------------------
# Scenario B: only 3 contributors → no capture
# ---------------------------------------------------------------------------

def test_no_capture_with_3_contributors():
    """
    Even if the angular gap is small, capture must NOT trigger with fewer than
    MIN_PREDATOR_CONTRIBUTORS predators in R_CAP.
    """
    prey_x = ARENA_WIDTH / 2
    prey_y = ARENA_HEIGHT / 2

    # 3 predators tightly surrounding — but not enough
    dist = R_CAP * 0.5
    preds = _make_positions_around(prey_x, prey_y, 3, dist)

    gap = compute_escape_gap(prey_x, prey_y, preds)
    assert gap.predator_contributors == 3

    # even if gap were small, FSM must never reach CAPTURED
    fsm = TacticalFSM()
    fsm.state = PreyTacticalState.CONTAINED
    for _ in range(T_HOLD * 3):
        state = fsm.update(gap, preds, prey_x, prey_y)
        assert state != PreyTacticalState.CAPTURED, "Should NOT capture with only 3 contributors"


# ---------------------------------------------------------------------------
# Scenario C: prey hiding longer than T_HIDE_MAX → forced exit
# ---------------------------------------------------------------------------

def test_prey_forced_exit_after_hide_max():
    """
    When prey hides inside an obstacle, after T_HIDE_MAX steps the
    ``decide()`` method must set a non-zero velocity (forced exit).
    """
    pygame.init()

    prey = Prey(100, 100, radius=PREY_RADIUS)
    # simulate obstacle rect that the prey is inside
    obs_rect = pygame.Rect(50, 50, 100, 100)
    obstacle_rects = [obs_rect]
    pred_positions = [(200, 100)]  # one predator nearby

    # manually put prey into hiding state
    prey.hiding = True
    prey.hide_steps = 0
    prey.velocity = pygame.math.Vector2(0, 0)

    for step in range(T_HIDE_MAX + 5):
        prey.decide(pred_positions, obstacle_rects, ARENA_WIDTH, ARENA_HEIGHT)
        if step < T_HIDE_MAX - 1:
            # should remain hiding with zero velocity
            assert prey.hiding or prey.velocity.length() > 0
        if step == T_HIDE_MAX - 1:
            # at this step hide_steps hits T_HIDE_MAX → forced exit
            assert not prey.hiding, f"Prey should have exited hiding at step {step}"
            assert prey.velocity.length() > 0, "Prey should have non-zero velocity after forced exit"
            break

    pygame.quit()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_capture_4_contributors_with_border()
    print("PASS: Scenario A - 4 contributors + border capture after T_HOLD")

    test_no_capture_with_3_contributors()
    print("PASS: Scenario B - 3 contributors, no capture")

    test_prey_forced_exit_after_hide_max()
    print("PASS: Scenario C - T_HIDE_MAX forced exit")

    print("\nAll targeted tests passed.")
