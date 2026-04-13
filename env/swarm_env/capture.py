"""
Angular capture geometry and tactical FSM.

One model, two callers: terminal capture check and prey escape-gap steering
both use ``compute_escape_gap``.  Obstacles are excluded from capture geometry
in V1 — only predator positions and arena borders participate.
"""

import math
from enum import Enum
from typing import NamedTuple

from swarm_env.config import (
    R_CAP,
    R_WALL_CAP,
    PHI_ESCAPE_MAX,
    PHI_CONTAINED,
    MARGIN_CONTAINED,
    R_DANGER,
    MARGIN_THREATENED,
    MIN_PREDATOR_CONTRIBUTORS,
    T_HOLD,
    ARENA_WIDTH,
    ARENA_HEIGHT,
)

TWO_PI = 2.0 * math.pi


# ── enums ─────────────────────────────────────────────────────────────────

class PreyTacticalState(Enum):
    FREE = 0
    THREATENED = 1
    CONTAINED = 2
    CAPTURED = 3


class EpisodeState(Enum):
    IN_PURSUIT = 0
    CAPTURED = 1
    TIMEOUT = 2


# ── gap result ────────────────────────────────────────────────────────────

class GapResult(NamedTuple):
    largest_gap: float          # radians
    gap_center_angle: float     # angle pointing into the center of that gap
    predator_contributors: int  # number of predators within R_CAP
    contributor_indices: list    # which predator indices are contributors
    border_blocker_count: int   # how many walls counted as blockers


# ── core geometry (single implementation) ─────────────────────────────────

def compute_escape_gap(
    prey_x: float,
    prey_y: float,
    predator_positions: list[tuple[float, float]],
    arena_w: float = ARENA_WIDTH,
    arena_h: float = ARENA_HEIGHT,
    r_cap: float = R_CAP,
    r_wall_cap: float = R_WALL_CAP,
) -> GapResult:
    """
    Compute the largest angular escape gap around the prey.

    Blockers are (a) predators within *r_cap* and (b) arena borders within
    *r_wall_cap*.  Each blocker is represented as a single angle from the
    prey's perspective.  The largest gap between consecutive blocker angles
    (sorted on the unit circle) is returned together with the center angle
    of that gap.

    Border angles (pygame coords — y increases downward):
        left wall   → π       right wall  → 0
        top wall    → −π/2    bottom wall → π/2
    """
    blocker_angles: list[float] = []
    contributor_indices: list[int] = []
    border_count = 0

    # predator contributors
    for i, (px, py) in enumerate(predator_positions):
        dx = px - prey_x
        dy = py - prey_y
        dist = math.hypot(dx, dy)
        if dist <= r_cap:
            blocker_angles.append(math.atan2(dy, dx))
            contributor_indices.append(i)

    # border blockers
    if prey_x <= r_wall_cap:                    # left wall
        blocker_angles.append(math.pi)
        border_count += 1
    if arena_w - prey_x <= r_wall_cap:          # right wall
        blocker_angles.append(0.0)
        border_count += 1
    if prey_y <= r_wall_cap:                    # top wall
        blocker_angles.append(-math.pi / 2)
        border_count += 1
    if arena_h - prey_y <= r_wall_cap:          # bottom wall
        blocker_angles.append(math.pi / 2)
        border_count += 1

    n_pred = len(contributor_indices)

    if not blocker_angles:
        return GapResult(TWO_PI, 0.0, n_pred, contributor_indices, border_count)

    if len(blocker_angles) == 1:
        a = blocker_angles[0]
        gap_center = a + math.pi  # opposite side
        gap_center = math.atan2(math.sin(gap_center), math.cos(gap_center))
        return GapResult(TWO_PI, gap_center, n_pred, contributor_indices, border_count)

    # sort and wrap
    blocker_angles.sort()
    max_gap = 0.0
    max_gap_center = 0.0

    for j in range(len(blocker_angles)):
        a1 = blocker_angles[j]
        a2 = blocker_angles[(j + 1) % len(blocker_angles)]
        gap = (a2 - a1) % TWO_PI
        if gap > max_gap:
            max_gap = gap
            center = a1 + gap / 2
            max_gap_center = math.atan2(math.sin(center), math.cos(center))

    return GapResult(max_gap, max_gap_center, n_pred, contributor_indices, border_count)


def nearest_predator_distance(
    prey_x: float,
    prey_y: float,
    predator_positions: list[tuple[float, float]],
) -> float:
    """Return distance to the closest predator, or inf if none."""
    best = float("inf")
    for px, py in predator_positions:
        d = math.hypot(px - prey_x, py - prey_y)
        if d < best:
            best = d
    return best


# ── tactical FSM (with hysteresis) ────────────────────────────────────────

class TacticalFSM:
    """
    Tracks prey tactical state (FREE → THREATENED → CONTAINED → CAPTURED)
    with configurable hysteresis margins to prevent flicker.
    """

    def __init__(self) -> None:
        self.state = PreyTacticalState.FREE
        self._hold_counter = 0

    def reset(self) -> None:
        self.state = PreyTacticalState.FREE
        self._hold_counter = 0

    def update(
        self,
        gap: GapResult,
        predator_positions: list[tuple[float, float]],
        prey_x: float,
        prey_y: float,
    ) -> PreyTacticalState:
        nearest = nearest_predator_distance(prey_x, prey_y, predator_positions)
        lg = gap.largest_gap

        if self.state == PreyTacticalState.FREE:
            if nearest <= R_DANGER:
                self.state = PreyTacticalState.THREATENED
                self._hold_counter = 0

        elif self.state == PreyTacticalState.THREATENED:
            if lg < PHI_CONTAINED:
                self.state = PreyTacticalState.CONTAINED
                self._hold_counter = 0
            elif nearest > R_DANGER + MARGIN_THREATENED:
                self.state = PreyTacticalState.FREE
                self._hold_counter = 0

        elif self.state == PreyTacticalState.CONTAINED:
            if lg > PHI_CONTAINED + MARGIN_CONTAINED:
                self.state = PreyTacticalState.THREATENED
                self._hold_counter = 0
            else:
                # temporary debug rule: if any predator is close enough, count as capture
                if nearest <= R_CAP:
                    self.state = PreyTacticalState.CAPTURED
                else:
                    self._hold_counter = 0

        return self.state
    