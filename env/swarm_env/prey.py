"""Bouncing-ball prey: constant-speed motion; arena walls reflect in ``Environment`` physics."""

import pygame

from swarm_env.config import DT, PREY_RADIUS, PREY_SPEED


class Prey:
    """
    Circular prey. Velocity is set at spawn; integration runs each step;
    the environment applies ``clamp_and_bounce`` on arena edges.
    """

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = PREY_RADIUS,
        speed: float = PREY_SPEED,
        vx: float = 0.0,
        vy: float = 0.0,
    ):
        self.position = pygame.math.Vector2(x, y)
        self.velocity = pygame.math.Vector2(vx, vy)
        self.collision_radius = radius
        self.speed = speed

    @property
    def radius(self) -> float:
        return self.collision_radius

    def get_collision_circle(self) -> tuple[tuple[float, float], float]:
        return ((self.position.x, self.position.y), self.collision_radius)

    def integrate(self, dt: float = DT) -> None:
        self.position += self.velocity * dt

    def draw(
        self,
        screen: pygame.Surface,
        fill: tuple = (220, 100, 90),
        outline: tuple = (255, 160, 140),
    ) -> None:
        cx = int(self.position.x)
        cy = int(self.position.y)
        r = int(self.collision_radius)
        pygame.draw.circle(screen, fill, (cx, cy), r)
        pygame.draw.circle(screen, outline, (cx, cy), r, 2)
