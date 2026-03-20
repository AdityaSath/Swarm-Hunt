"""Hexagon-shaped drone: independent agent with limited perception."""

import math

import pygame

from swarm_env.config import (
    DRONE_RADIUS,
    DRONE_VISION_ANGLE_DEG,
    DRONE_VISION_RANGE,
    DRONE_PERCEPTION_RANGE,
)


class Drone:
    """
    Independent agent with local state only. Does not read full world state.

    Attributes:
        position: (x, y) in world coordinates
        velocity: (vx, vy) in world coordinates
        heading: facing angle in radians (0 = right, pi/2 = down)
        collision_radius: radius for collision detection
        perception_range: max distance to sense obstacles and other drones
        vision_angle_deg: half-angle of vision cone for display (degrees)
        vision_range: length of vision cone for display
    """

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = DRONE_RADIUS,
        vx: float = 0.0,
        vy: float = 0.0,
        perception_range: float = DRONE_PERCEPTION_RANGE,
        vision_angle_deg: float = DRONE_VISION_ANGLE_DEG,
        vision_range: float = DRONE_VISION_RANGE,
    ):
        self.position = pygame.math.Vector2(x, y)
        self.velocity = pygame.math.Vector2(vx, vy)
        self.heading = math.atan2(vy, vx) if (vx * vx + vy * vy) > 0 else 0.0
        self.collision_radius = radius
        self.perception_range = perception_range
        self.vision_angle_deg = vision_angle_deg
        self.vision_range = vision_range

    @property
    def radius(self) -> float:
        """Alias for collision_radius (backward compatibility)."""
        return self.collision_radius

    def get_vertices(self) -> list[tuple[float, float]]:
        """Return list of (x, y) vertices for drawing hexagon (oriented by heading)."""
        cx, cy = self.position.x, self.position.y
        points = []
        for i in range(6):
            angle = self.heading + math.pi / 3 * i - math.pi / 6
            px = cx + self.collision_radius * math.cos(angle)
            py = cy + self.collision_radius * math.sin(angle)
            points.append((px, py))
        return points

    def get_collision_circle(self) -> tuple[tuple[float, float], float]:
        """Return (center, radius) for collision detection."""
        return ((self.position.x, self.position.y), self.collision_radius)

    def update(self, dt: float) -> None:
        """Apply velocity to position. Updates heading from velocity. Caller handles collision."""
        self.position += self.velocity * dt
        if self.velocity.length_squared() > 0:
            self.heading = math.atan2(self.velocity.y, self.velocity.x)

    def get_vision_cone_vertices(self) -> list[tuple[float, float]]:
        """Return vertices for the vision cone: [center, left_edge, right_edge]."""
        cx, cy = self.position.x, self.position.y
        half_rad = math.radians(self.vision_angle_deg)
        left_angle = self.heading - half_rad
        right_angle = self.heading + half_rad
        left_x = cx + self.vision_range * math.cos(left_angle)
        left_y = cy + self.vision_range * math.sin(left_angle)
        right_x = cx + self.vision_range * math.cos(right_angle)
        right_y = cy + self.vision_range * math.sin(right_angle)
        return [(cx, cy), (left_x, left_y), (right_x, right_y)]

    def draw(self, screen: pygame.Surface, color: tuple = (60, 140, 200)):
        """Draw the vision cone, then the hexagon drone."""
        cone_points = self.get_vision_cone_vertices()
        cone_surf = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
        pygame.draw.polygon(cone_surf, (60, 140, 200, 60), cone_points)
        screen.blit(cone_surf, (0, 0))
        pygame.draw.polygon(screen, (80, 160, 220), cone_points, 1)

        points = self.get_vertices()
        pygame.draw.polygon(screen, color, points)
        pygame.draw.polygon(screen, (100, 180, 220), points, 1)
