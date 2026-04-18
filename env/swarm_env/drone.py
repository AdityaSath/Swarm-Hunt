"""Predator drone: desired-velocity dynamics, local perception."""

import pygame

from swarm_env.config import DRONE_RADIUS, DRONE_SPEED, R_SENSE, DT


class Drone:
    """
    Predator agent with desired-velocity control.

    Each step the environment provides (vx_desired, vy_desired).  The drone
    keeps the command direction and clips the magnitude to DRONE_SPEED, so
    the policy can decide both heading **and** speed (including stopping).

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
        """Set velocity from desired (vx, vy); clip magnitude to DRONE_SPEED."""
        v = pygame.math.Vector2(vx, vy)
        if v.length_squared() > DRONE_SPEED * DRONE_SPEED:
            v.scale_to_length(DRONE_SPEED)
        self.velocity = v

    def integrate(self, dt: float = DT) -> None:
        """Advance position by one timestep. Caller handles collision."""
        self.position += self.velocity * dt

    # ----- rendering helpers -----

    def get_collision_circle(self) -> tuple[tuple[float, float], float]:
        return ((self.position.x, self.position.y), self.collision_radius)

    def draw(self, screen: pygame.Surface, color: tuple = (60, 140, 200)):
        cx = int(round(self.position.x))
        cy = int(round(self.position.y))
        r = max(1, int(round(self.collision_radius)))
        pygame.draw.circle(screen, color, (cx, cy), r)
        pygame.draw.circle(screen, (100, 180, 220), (cx, cy), r, 1)
