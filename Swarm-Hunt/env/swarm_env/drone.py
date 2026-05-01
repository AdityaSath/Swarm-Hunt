"""Predator drone: desired-velocity dynamics, local perception."""

import math

import pygame

from swarm_env.config import DRONE_RADIUS, DRONE_SPEED, R_SENSE, DT


class Drone:
    """
    Predator agent with desired-velocity control.

    Each step the environment provides (vx_desired, vy_desired).  The drone
    clips the magnitude to DRONE_SPEED and integrates position by DT.

    Attributes:
        position: world (x, y)
        velocity: world (vx, vy) after clipping
        collision_radius: radius for physics
        perception_range: R_SENSE from config
    """

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = DRONE_RADIUS,
        vx: float = 0.0,
        vy: float = 0.0,
        perception_range: float = R_SENSE,
    ):
        self.position = pygame.math.Vector2(x, y)
        self.velocity = pygame.math.Vector2(vx, vy)
        self.collision_radius = radius
        self.perception_range = perception_range

    @property
    def radius(self) -> float:
        return self.collision_radius

    def set_desired_velocity(self, vx: float, vy: float) -> None:
        """Set velocity from desired (vx, vy), clipping magnitude to DRONE_SPEED."""
        v = pygame.math.Vector2(vx, vy)
        if v.length_squared() > DRONE_SPEED * DRONE_SPEED:
            v.scale_to_length(DRONE_SPEED)
        self.velocity = v

    def integrate(self, dt: float = DT) -> None:
        """Advance position by one timestep. Caller handles collision."""
        self.position += self.velocity * dt

    # ----- rendering helpers -----

    @property
    def heading(self) -> float:
        """Visual heading from current velocity (0 when stationary)."""
        if self.velocity.length_squared() > 0:
            return math.atan2(self.velocity.y, self.velocity.x)
        return 0.0

    def get_vertices(self) -> list[tuple[float, float]]:
        cx, cy = self.position.x, self.position.y
        r = self.collision_radius
        h = self.heading
        cos_h = math.cos(h)
        sin_h = math.sin(h)
        tip = (cx + r * cos_h, cy + r * sin_h)
        back_left = (cx - 0.7 * r * cos_h + 0.4 * r * sin_h,
                     cy - 0.7 * r * sin_h - 0.4 * r * cos_h)
        back_right = (cx - 0.7 * r * cos_h - 0.4 * r * sin_h,
                      cy - 0.7 * r * sin_h + 0.4 * r * cos_h)
        return [tip, back_left, back_right]

    def get_collision_circle(self) -> tuple[tuple[float, float], float]:
        return ((self.position.x, self.position.y), self.collision_radius)

    def draw(self, screen: pygame.Surface, color: tuple = (60, 140, 200)):
        points = self.get_vertices()
        pygame.draw.polygon(screen, color, points)
        pygame.draw.polygon(screen, (100, 180, 220), points, 1)
