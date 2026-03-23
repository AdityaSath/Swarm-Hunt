"""Arrow-shaped drone: independent agent with limited perception."""

import math

import pygame

from swarm_env.config import (
    DRONE_RADIUS,
    DRONE_SPEED,
    DRONE_MAX_TURN_RATE,
    DRONE_PERCEPTION_RANGE,
)


class Drone:
    """
    Independent agent with local state only. Does not read full world state.

    Helicopter-style movement: thrust (forward/back) + steer (turn left/right).
    - Thrust only: move straight. Steer only: spin in place.
    - Thrust + steer: curved path (circle).

    Attributes:
        position: (x, y) in world coordinates
        velocity: (vx, vy) in world coordinates
        heading: facing angle in radians (0 = right, pi/2 = down)
        thrust: -1 to 1 (backward to forward)
        steer: -1 to 1 (turn left to turn right)
        collision_radius: radius for collision detection
        perception_range: max distance to sense obstacles and other drones
    """

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = DRONE_RADIUS,
        vx: float = 0.0,
        vy: float = 0.0,
        perception_range: float = DRONE_PERCEPTION_RANGE,
    ):
        self.position = pygame.math.Vector2(x, y)
        self.velocity = pygame.math.Vector2(vx, vy)
        self.heading = math.atan2(vy, vx) if (vx * vx + vy * vy) > 0 else 0.0
        # Thrust/steer: init from velocity for spawn; will be set by actions each step
        speed = math.hypot(vx, vy)
        self.thrust = speed / DRONE_SPEED if DRONE_SPEED > 0 else 0.0
        self.thrust = max(-1.0, min(1.0, self.thrust))
        self.steer = 0.0
        self.collision_radius = radius
        self.perception_range = perception_range

    @property
    def radius(self) -> float:
        """Alias for collision_radius (backward compatibility)."""
        return self.collision_radius

    def get_vertices(self) -> list[tuple[float, float]]:
        """Return list of (x, y) vertices for drawing arrow (pointing in heading direction)."""
        cx, cy = self.position.x, self.position.y
        r = self.collision_radius
        h = self.heading
        cos_h = math.cos(h)
        sin_h = math.sin(h)
        # Tip (front), back-left, back-right
        tip = (cx + r * cos_h, cy + r * sin_h)
        back_left = (cx - 0.7 * r * cos_h + 0.4 * r * sin_h, cy - 0.7 * r * sin_h - 0.4 * r * cos_h)
        back_right = (cx - 0.7 * r * cos_h - 0.4 * r * sin_h, cy - 0.7 * r * sin_h + 0.4 * r * cos_h)
        return [tip, back_left, back_right]

    def get_collision_circle(self) -> tuple[tuple[float, float], float]:
        """Return (center, radius) for collision detection."""
        return ((self.position.x, self.position.y), self.collision_radius)

    def update(self, dt: float) -> None:
        """Helicopter-style: steer changes heading, thrust sets velocity along heading. Caller handles collision."""
        self.heading += self.steer * DRONE_MAX_TURN_RATE * dt
        speed = self.thrust * DRONE_SPEED
        self.velocity = pygame.math.Vector2(
            speed * math.cos(self.heading),
            speed * math.sin(self.heading),
        )
        self.position += self.velocity * dt

    def draw(self, screen: pygame.Surface, color: tuple = (60, 140, 200)):
        """Draw the arrow-shaped drone."""
        points = self.get_vertices()
        pygame.draw.polygon(screen, color, points)
        pygame.draw.polygon(screen, (100, 180, 220), points, 1)
