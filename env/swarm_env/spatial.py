"""
Spatial neighbor lookup. Pluggable implementations for efficient neighbor search.

Start with distance-based O(n^2); structure allows swapping to grid/quadtree later.
"""

from typing import Protocol

import pygame


class NeighborFinder(Protocol):
    """Protocol for spatial neighbor lookup. Swap implementation for grid/quadtree later."""

    def find_drone_neighbors(
        self,
        positions: list[tuple[float, float]],
        radius: float,
        exclude_self: bool = True,
    ) -> list[list[int]]:
        """
        For each position, return indices of other positions within radius.
        Returns list of length len(positions); each element is list of neighbor indices.
        """
        ...

    def find_nearby_obstacles(
        self,
        query_positions: list[tuple[float, float]],
        obstacle_rects: list[pygame.Rect],
        radius: float,
    ) -> list[list[tuple[int, float, tuple[float, float]]]]:
        """
        For each query position, return obstacles within radius.
        Returns list of (obstacle_idx, distance, relative_pos) per query.
        """
        ...


class DistanceBasedNeighborFinder:
    """Simple O(n^2) distance-based neighbor search. Replace with GridNeighborFinder for scale."""

    def find_drone_neighbors(
        self,
        positions: list[tuple[float, float]],
        radius: float,
        exclude_self: bool = True,
    ) -> list[list[int]]:
        r_sq = radius * radius
        result: list[list[int]] = [[] for _ in positions]
        for i, (px, py) in enumerate(positions):
            for j, (qx, qy) in enumerate(positions):
                if exclude_self and i == j:
                    continue
                dx = qx - px
                dy = qy - py
                if dx * dx + dy * dy <= r_sq:
                    result[i].append(j)
        return result

    def find_nearby_obstacles(
        self,
        query_positions: list[tuple[float, float]],
        obstacle_rects: list[pygame.Rect],
        radius: float,
    ) -> list[list[tuple[int, float, tuple[float, float]]]]:
        import math

        r_sq = radius * radius
        result: list[list[tuple[int, float, tuple[float, float]]]] = [[] for _ in query_positions]
        for qi, (px, py) in enumerate(query_positions):
            for oi, rect in enumerate(obstacle_rects):
                cx = max(rect.left, min(rect.right, px))
                cy = max(rect.top, min(rect.bottom, py))
                dx = px - cx
                dy = py - cy
                dist_sq = dx * dx + dy * dy
                if dist_sq <= r_sq:
                    dist = math.sqrt(dist_sq)
                    result[qi].append((oi, dist, (dx, dy)))
        return result
