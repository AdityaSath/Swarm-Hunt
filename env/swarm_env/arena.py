"""Bounded arena for the swarm environment."""

import pygame.math

from swarm_env.config import ARENA_WIDTH, ARENA_HEIGHT


class Arena:
    """Bounded rectangle arena with boundary logic."""

    def __init__(self, width: int = ARENA_WIDTH, height: int = ARENA_HEIGHT):
        self.width = width
        self.height = height
        self._x_min = 0
        self._y_min = 0
        self._x_max = width
        self._y_max = height

    def contains(self, point: pygame.math.Vector2) -> bool:
        """Check if point is inside bounds."""
        return (
            self._x_min <= point.x <= self._x_max
            and self._y_min <= point.y <= self._y_max
        )

    def clamp(
        self, position: pygame.math.Vector2, margin: float = 0.0
    ) -> pygame.math.Vector2:
        """Return position clamped to valid region with optional margin for drone radius."""
        x_min = self._x_min + margin
        y_min = self._y_min + margin
        x_max = self._x_max - margin
        y_max = self._y_max - margin
        x = max(x_min, min(x_max, position.x))
        y = max(y_min, min(y_max, position.y))
        return pygame.math.Vector2(x, y)

    def clamp_and_bounce(
        self,
        position: pygame.math.Vector2,
        velocity: pygame.math.Vector2,
        margin: float = 0.0,
    ) -> tuple[pygame.math.Vector2, pygame.math.Vector2]:
        """Clamp position to bounds and reflect velocity when hitting walls. Returns (new_pos, new_vel)."""
        x_min = self._x_min + margin
        y_min = self._y_min + margin
        x_max = self._x_max - margin
        y_max = self._y_max - margin
        vx, vy = velocity.x, velocity.y
        x, y = position.x, position.y

        if x < x_min:
            x = x_min
            vx = abs(vx)
        elif x > x_max:
            x = x_max
            vx = -abs(vx)
        if y < y_min:
            y = y_min
            vy = abs(vy)
        elif y > y_max:
            y = y_max
            vy = -abs(vy)

        return (
            pygame.math.Vector2(x, y),
            pygame.math.Vector2(vx, vy),
        )

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) for algorithms."""
        return (self._x_min, self._y_min, self._x_max, self._y_max)

    def draw(self, screen: pygame.Surface, color: tuple = (40, 40, 50)):
        """Draw arena border and background."""
        rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (100, 100, 120), rect, 2)
