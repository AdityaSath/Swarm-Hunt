"""
Distance-based capture and tactical FSM.

Capture hold: each step, count **arena walls intersecting** the ``R_CAPTURE_RANGE``
disk around the prey, plus **predators** whose centers are inside that disk.
If ``walls + drones >= COMBO_CAPTURE_NEED`` for ``CAPTURE_HOLD_STEPS`` consecutive
steps → CAPTURED.

Threat: FREE ↔ THREATENED uses ``R_DANGER`` (nearest-predator distance).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import NamedTuple

from swarm_env.config import (
    R_CAPTURE_RANGE,
    R_DANGER,
    MARGIN_THREATENED,
    COMBO_CAPTURE_NEED,
    CAPTURE_HOLD_STEPS,
)


# ── enums ─────────────────────────────────────────────────────────────────


class PreyTacticalState(Enum):
    FREE = 0
    THREATENED = 1
    CAPTURED = 2


class EpisodeState(Enum):
    IN_PURSUIT = 0
    CAPTURED = 1
    TIMEOUT = 2


class CaptureStatus(NamedTuple):
    """Published in env infos each step."""

    in_range_count: int
    contributor_indices: list[int]
    hold_counter: int
    wall_count: int  # arena edges (0–4) intersecting the R_CAPTURE_RANGE disk


# ── geometry helpers ──────────────────────────────────────────────────────


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


def predators_in_capture_range(
    prey_x: float,
    prey_y: float,
    predator_positions: list[tuple[float, float]],
    radius: float = R_CAPTURE_RANGE,
) -> tuple[int, list[int]]:
    """Count predators whose center is within *radius* of prey; return indices."""
    indices: list[int] = []
    for i, (px, py) in enumerate(predator_positions):
        if math.hypot(px - prey_x, py - prey_y) <= radius:
            indices.append(i)
    return len(indices), indices


def walls_intersecting_capture_circle(
    prey_x: float,
    prey_y: float,
    arena_w: float,
    arena_h: float,
    circle_r: float = R_CAPTURE_RANGE,
) -> int:
    """
    Count rectangular arena edges whose perpendicular distance to the prey
    center is at most *circle_r* (the blue capture disk reaches that wall).
    """
    n = 0
    if prey_x <= circle_r:
        n += 1
    if arena_w - prey_x <= circle_r:
        n += 1
    if prey_y <= circle_r:
        n += 1
    if arena_h - prey_y <= circle_r:
        n += 1
    return n


def flee_angle_from_nearest_predator(
    prey_x: float,
    prey_y: float,
    predator_positions: list[tuple[float, float]],
) -> float | None:
    """Unit direction *away* from nearest predator, as atan2 angle; None if empty."""
    best_d = float("inf")
    best: tuple[float, float] | None = None
    for px, py in predator_positions:
        d = math.hypot(px - prey_x, py - prey_y)
        if d < best_d:
            best_d = d
            best = (px, py)
    if best is None:
        return None
    dx = prey_x - best[0]
    dy = prey_y - best[1]
    if dx == 0.0 and dy == 0.0:
        return None
    return math.atan2(dy, dx)


# ── tactical FSM ──────────────────────────────────────────────────────────


class TacticalFSM:
    """
    FREE ↔ THREATENED (``R_DANGER``).  CAPTURED when
    ``walls_intersecting_capture_circle + drones_in_R_CAPTURE_RANGE >= COMBO_CAPTURE_NEED``
    for ``CAPTURE_HOLD_STEPS`` consecutive steps.
    """

    def __init__(
        self,
        combo_capture_need: int = COMBO_CAPTURE_NEED,
        capture_hold_steps: int = CAPTURE_HOLD_STEPS,
    ) -> None:
        self.state = PreyTacticalState.FREE
        self._combo_capture_need = combo_capture_need
        self._capture_hold_steps = max(1, int(capture_hold_steps))
        self._hold_counter = 0
        self._last_wall_count = 0

    def reset(self) -> None:
        self.state = PreyTacticalState.FREE
        self._hold_counter = 0
        self._last_wall_count = 0

    @property
    def hold_counter(self) -> int:
        return self._hold_counter

    @property
    def last_wall_count(self) -> int:
        return self._last_wall_count

    def update(
        self,
        predator_positions: list[tuple[float, float]],
        prey_x: float,
        prey_y: float,
        arena_w: float,
        arena_h: float,
    ) -> PreyTacticalState:
        if self.state == PreyTacticalState.CAPTURED:
            return self.state

        n_in, _ = predators_in_capture_range(prey_x, prey_y, predator_positions)
        w = walls_intersecting_capture_circle(
            prey_x, prey_y, arena_w, arena_h, R_CAPTURE_RANGE,
        )
        self._last_wall_count = w
        combined = w + n_in
        qualifying = combined >= self._combo_capture_need

        if qualifying:
            self._hold_counter += 1
        else:
            self._hold_counter = 0

        if self._hold_counter >= self._capture_hold_steps:
            self.state = PreyTacticalState.CAPTURED
            return self.state

        nearest = nearest_predator_distance(prey_x, prey_y, predator_positions)

        if self.state == PreyTacticalState.FREE:
            if nearest <= R_DANGER:
                self.state = PreyTacticalState.THREATENED
        elif self.state == PreyTacticalState.THREATENED:
            if nearest > R_DANGER + MARGIN_THREATENED:
                self.state = PreyTacticalState.FREE

        return self.state
