"""Predator drone with acceleration-limited desired-velocity dynamics."""

import pygame

from swarm_env.config import (
    DRONE_RADIUS,
    DRONE_SPEED,
    DRONE_MAX_ACCELERATION,
    R_SENSE,
    DT,
)


class Drone:
    """
    Predator agent with desired-velocity control.

    Each step the environment provides (vx_desired, vy_desired). The command
    is speed-clipped, then the actual velocity approaches it at no more than
    DRONE_MAX_ACCELERATION. This avoids instantaneous direction reversals
    while preserving direct velocity control for the policy.

    Attributes:
        position: world (x, y)
        velocity: actual world velocity (vx, vy)
        desired_velocity: clipped policy command (vx, vy)
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
        initial_velocity = pygame.math.Vector2(vx, vy)
        if initial_velocity.length_squared() > DRONE_SPEED * DRONE_SPEED:
            initial_velocity.scale_to_length(DRONE_SPEED)
        self.velocity = initial_velocity
        self.desired_velocity = initial_velocity.copy()
        self.collision_radius = radius
        self.perception_range = perception_range

    @property
    def radius(self) -> float:
        return self.collision_radius

    def set_desired_velocity(self, vx: float, vy: float) -> None:
        """Store a desired velocity, clipped to the configured top speed."""
        v = pygame.math.Vector2(vx, vy)
        if v.length_squared() > DRONE_SPEED * DRONE_SPEED:
            v.scale_to_length(DRONE_SPEED)
        self.desired_velocity = v

    def integrate(self, dt: float = DT) -> None:
        """Approach the command at bounded acceleration, then advance position."""
        velocity_delta = self.desired_velocity - self.velocity
        max_delta = max(0.0, DRONE_MAX_ACCELERATION * dt)
        if velocity_delta.length_squared() > max_delta * max_delta:
            velocity_delta.scale_to_length(max_delta)
        self.velocity += velocity_delta

        # Numerical guard: acceleration limiting should not exceed this, but
        # collision response and future dynamics should never violate max speed.
        if self.velocity.length_squared() > DRONE_SPEED * DRONE_SPEED:
            self.velocity.scale_to_length(DRONE_SPEED)
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
