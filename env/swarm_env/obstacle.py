"""Static obstacles for the swarm environment."""

import pygame

from swarm_env.config import OBSTACLE_SIZES


class Obstacle:
    """Static rectangular obstacle."""

    def __init__(self, x: float, y: float, size_type: str = "small"):
        w, h = OBSTACLE_SIZES.get(size_type, OBSTACLE_SIZES["small"])
        self.rect = pygame.Rect(0, 0, w, h)
        self.rect.center = (x, y)
        self.size_type = size_type

    def get_collision_rect(self) -> pygame.Rect:
        """Return rect for collision checks."""
        return self.rect

    def draw(self, screen: pygame.Surface, color: tuple = (120, 80, 80)):
        """Draw the obstacle."""
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (160, 100, 100), self.rect, 1)
