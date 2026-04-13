"""
Scripted prey: desired-velocity dynamics, obstacle hiding, distance-based evasion.

The prey is NOT a learning agent in V1.  Its policy is entirely rule-based:
    1. If threatened → flee away from the nearest predator.
    2. If threatened and an obstacle is nearby → may enter it.
    3. While hidden inside an obstacle → stay until T_HIDE_MAX, then exit.
    4. Otherwise → flee from nearest sensed predator cluster.
"""

import math
import random

import pygame

from swarm_env.config import (
    PREY_RADIUS,
    PREY_SPEED,
    R_SENSE,
    R_DANGER,
    T_HIDE_MAX,
    DT,
)
from swarm_env.capture import flee_angle_from_nearest_predator, nearest_predator_distance


class Prey:
    """
    Circular prey with desired-velocity control and obstacle-hiding state.
    """

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = PREY_RADIUS,
        vx: float = 0.0,
        vy: float = 0.0,
        speed: float = PREY_SPEED,
    ):
        self.position = pygame.math.Vector2(x, y)
        self.velocity = pygame.math.Vector2(vx, vy)
        self.collision_radius = radius
        self.speed = speed

        # hide state
        self.hiding = False
        self.hide_steps = 0

    @property
    def radius(self) -> float:
        return self.collision_radius

    def get_collision_circle(self) -> tuple[tuple[float, float], float]:
        return ((self.position.x, self.position.y), self.collision_radius)

    # ── hiding helpers ────────────────────────────────────────────────────

    def is_inside_any_obstacle(self, obstacle_rects: list[pygame.Rect]) -> bool:
        """True if prey center is inside any obstacle rect."""
        px, py = self.position.x, self.position.y
        for rect in obstacle_rects:
            if rect.collidepoint(px, py):
                return True
        return False

    def _nearest_obstacle_center(
        self, obstacle_rects: list[pygame.Rect],
    ) -> tuple[float, float] | None:
        """Return center of the closest obstacle, or None."""
        best_d = float("inf")
        best_c = None
        px, py = self.position.x, self.position.y
        for rect in obstacle_rects:
            cx, cy = rect.centerx, rect.centery
            d = math.hypot(cx - px, cy - py)
            if d < best_d:
                best_d = d
                best_c = (cx, cy)
        return best_c

    # ── scripted policy ───────────────────────────────────────────────────

    def decide(
        self,
        predator_positions: list[tuple[float, float]],
        obstacle_rects: list[pygame.Rect],
        arena_w: float,
        arena_h: float,
    ) -> None:
        """
        Set ``self.velocity`` according to the scripted evasion policy.
        Called once per env step, *before* integration.
        """
        px, py = self.position.x, self.position.y

        # --- hiding logic -------------------------------------------------
        inside = self.is_inside_any_obstacle(obstacle_rects)

        if self.hiding:
            self.hide_steps += 1
            if self.hide_steps >= T_HIDE_MAX:
                # forced exit: flee from nearest predator
                self.hiding = False
                self.hide_steps = 0
                ang = flee_angle_from_nearest_predator(px, py, predator_positions)
                if ang is not None:
                    self._set_velocity_toward(ang)
                return
            if inside:
                # stay still while hidden and timer hasn't elapsed
                self.velocity = pygame.math.Vector2(0, 0)
                return
            else:
                # left the obstacle (pushed by wall clamp etc.) — no longer hiding
                self.hiding = False
                self.hide_steps = 0

        # --- threat assessment --------------------------------------------
        nearest = nearest_predator_distance(px, py, predator_positions)
        threatened = nearest <= R_DANGER

        if threatened:
            # consider entering a nearby obstacle to hide
            obs_center = self._nearest_obstacle_center(obstacle_rects)
            if obs_center is not None:
                dx = obs_center[0] - px
                dy = obs_center[1] - py
                obs_dist = math.hypot(dx, dy)
                if obs_dist < R_SENSE * 0.4 and not inside:
                    # move toward obstacle to hide
                    self._set_velocity_toward(math.atan2(dy, dx))
                    return
                if inside and not self.hiding:
                    self.hiding = True
                    self.hide_steps = 0
                    self.velocity = pygame.math.Vector2(0, 0)
                    return

            ang = flee_angle_from_nearest_predator(px, py, predator_positions)
            if ang is not None:
                self._set_velocity_toward(ang)
            return

        # --- not threatened: gentle wander away from nearest cluster ------
        if predator_positions:
            avg_x = sum(p[0] for p in predator_positions) / len(predator_positions)
            avg_y = sum(p[1] for p in predator_positions) / len(predator_positions)
            dx = px - avg_x
            dy = py - avg_y
            dist = math.hypot(dx, dy)
            if dist > 0:
                angle = math.atan2(dy, dx)
                # add slight randomness to avoid deterministic loops
                angle += random.uniform(-0.3, 0.3)
                self._set_velocity_toward(angle, speed_frac=0.6)
                return

        # fallback: drift
        if self.velocity.length_squared() < 1.0:
            angle = random.uniform(-math.pi, math.pi)
            self._set_velocity_toward(angle, speed_frac=0.4)

    # ── helpers ───────────────────────────────────────────────────────────

    def _set_velocity_toward(self, angle: float, speed_frac: float = 1.0) -> None:
        speed = self.speed * max(0.0, min(1.0, speed_frac))
        self.velocity = pygame.math.Vector2(
            speed * math.cos(angle),
            speed * math.sin(angle),
        )

    def integrate(self, dt: float = DT) -> None:
        """Advance position. Caller handles arena clamp (prey ignores obstacles)."""
        self.position += self.velocity * dt

    # ── rendering ─────────────────────────────────────────────────────────

    def draw(
        self,
        screen: pygame.Surface,
        fill: tuple = (220, 100, 90),
        outline: tuple = (255, 160, 140),
    ):
        cx = int(self.position.x)
        cy = int(self.position.y)
        r = int(self.collision_radius)
        pygame.draw.circle(screen, fill, (cx, cy), r)
        pygame.draw.circle(screen, outline, (cx, cy), r, 2)
