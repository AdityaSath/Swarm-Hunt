"""
Main swarm environment. RL-ready: reset, step(actions), observation-based updates.

Environment logic is separate from drone logic. Drones are state containers;
the environment applies actions, physics, and provides observations via local neighbor lookup.
"""

import math
import random
from typing import Any

import pygame

from swarm_env.arena import Arena
from swarm_env.obstacle import Obstacle
from swarm_env.drone import Drone
from swarm_env.spatial import DistanceBasedNeighborFinder, NeighborFinder
from swarm_env.config import (
    ARENA_WIDTH,
    ARENA_HEIGHT,
    DRONE_COUNT,
    DRONE_RADIUS,
    DRONE_SPEED,
    OBSTACLE_POSITIONS,
    OBSTACLE_REPULSION_RANGE,
    OBSTACLE_REPULSION_STRENGTH,
    DT,
)


def _obstacle_repulsion(
    position: pygame.math.Vector2,
    obstacles: list,
) -> pygame.math.Vector2:
    """Return repulsion velocity (1/distance) from nearby obstacles."""
    total = pygame.math.Vector2(0, 0)
    px, py = position.x, position.y

    for obs in obstacles:
        rect = obs.get_collision_rect()
        closest_x = max(rect.left, min(rect.right, px))
        closest_y = max(rect.top, min(rect.bottom, py))
        dx = px - closest_x
        dy = py - closest_y
        dist_sq = dx * dx + dy * dy
        dist = math.sqrt(dist_sq)

        if dist <= 0 or dist > OBSTACLE_REPULSION_RANGE:
            continue

        # 1/distance falloff; clamp min dist to avoid explosion when very close
        safe_dist = max(dist, 5.0)
        magnitude = OBSTACLE_REPULSION_STRENGTH / (safe_dist * safe_dist)
        nx = dx / safe_dist
        ny = dy / safe_dist
        total.x += nx * magnitude
        total.y += ny * magnitude

    return total


def _circle_rect_overlap(
    center: tuple[float, float], radius: float, rect: pygame.Rect
) -> bool:
    """Check if circle overlaps rect."""
    cx, cy = center
    closest_x = max(rect.left, min(rect.right, cx))
    closest_y = max(rect.top, min(rect.bottom, cy))
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy < radius * radius


def _push_circle_out_of_rect(
    center: pygame.math.Vector2, radius: float, rect: pygame.Rect
) -> tuple[pygame.math.Vector2, pygame.math.Vector2 | None]:
    """Push circle center out of rect. Returns (new_position, collision_normal or None)."""
    cx, cy = center.x, center.y
    closest_x = max(rect.left, min(rect.right, cx))
    closest_y = max(rect.top, min(rect.bottom, cy))
    dx = cx - closest_x
    dy = cy - closest_y
    dist_sq = dx * dx + dy * dy

    if dist_sq >= radius * radius:
        return (center, None)

    if dist_sq == 0:
        to_left = cx - rect.left
        to_right = rect.right - cx
        to_top = cy - rect.top
        to_bottom = rect.bottom - cy
        min_dist = min(to_left, to_right, to_top, to_bottom)
        if min_dist == to_left:
            return (pygame.math.Vector2(rect.left - radius, cy), pygame.math.Vector2(1, 0))
        if min_dist == to_right:
            return (pygame.math.Vector2(rect.right + radius, cy), pygame.math.Vector2(-1, 0))
        if min_dist == to_top:
            return (pygame.math.Vector2(cx, rect.top - radius), pygame.math.Vector2(0, 1))
        return (pygame.math.Vector2(cx, rect.bottom + radius), pygame.math.Vector2(0, -1))

    dist = math.sqrt(dist_sq)
    overlap = radius - dist
    nx = dx / dist
    ny = dy / dist
    normal = pygame.math.Vector2(nx, ny)
    return (center + normal * overlap, normal)


def _circle_circle_overlap(
    c1: tuple[float, float], r1: float, c2: tuple[float, float], r2: float
) -> bool:
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return dx * dx + dy * dy < (r1 + r2) ** 2


def _push_circles_apart(
    pos1: pygame.math.Vector2,
    r1: float,
    pos2: pygame.math.Vector2,
    r2: float,
) -> tuple[pygame.math.Vector2, pygame.math.Vector2]:
    dx = pos2.x - pos1.x
    dy = pos2.y - pos1.y
    dist_sq = dx * dx + dy * dy
    if dist_sq == 0:
        pos1 = pos1 + pygame.math.Vector2(r1, 0)
        pos2 = pos2 + pygame.math.Vector2(-r2, 0)
        return (pos1, pos2)
    dist = math.sqrt(dist_sq)
    overlap = r1 + r2 - dist
    if overlap <= 0:
        return (pos1, pos2)
    nx = dx / dist
    ny = dy / dist
    half = overlap / 2
    pos1 = pos1 - pygame.math.Vector2(nx * half, ny * half)
    pos2 = pos2 + pygame.math.Vector2(nx * half, ny * half)
    return (pos1, pos2)


class Environment:
    """
    RL-ready swarm environment. Separate environment logic from drone logic.

    API:
        reset(seed) -> observations, infos
        step(actions) -> observations, rewards, terminations, truncations, infos
        get_observation(agent_id) -> local observation (uses neighbor finder)
    """

    def __init__(
        self,
        width: int = ARENA_WIDTH,
        height: int = ARENA_HEIGHT,
        drone_count: int = DRONE_COUNT,
        dt: float = DT,
        seed: int | None = None,
        neighbor_finder: NeighborFinder | None = None,
    ):
        if seed is not None:
            random.seed(seed)
        self.dt = dt
        self.arena = Arena(width, height)
        self.neighbor_finder = neighbor_finder or DistanceBasedNeighborFinder()
        self.obstacles: list[Obstacle] = []
        self.drones: list[Drone] = []
        self._drone_count = drone_count
        self._seed = seed
        self._width = width
        self._height = height
        self._step_count = 0
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[dict[int, dict], dict[str, Any]]:
        """
        Reset environment. Returns (observations, infos).
        Compatible with PettingZoo-style reset.
        """
        if seed is not None:
            random.seed(seed)
        self._step_count = 0
        self.obstacles.clear()
        self.drones.clear()
        self._init_obstacles()
        self._init_drones(self._drone_count)
        observations = self._compute_observations()
        infos = {"agents": list(range(len(self.drones)))}
        return observations, infos

    def _init_obstacles(self) -> None:
        for x, y, size_type in OBSTACLE_POSITIONS:
            self.obstacles.append(Obstacle(x, y, size_type))

    def _is_valid_spawn(self, x: float, y: float, margin: float) -> bool:
        if x - margin < 0 or x + margin > self.arena.width:
            return False
        if y - margin < 0 or y + margin > self.arena.height:
            return False
        test_rect = pygame.Rect(x - margin, y - margin, 2 * margin, 2 * margin)
        for obs in self.obstacles:
            if test_rect.colliderect(obs.rect):
                return False
        return True

    def _init_drones(self, count: int) -> None:
        margin = DRONE_RADIUS * 2.5
        attempts = 0
        max_attempts = count * 100
        while len(self.drones) < count and attempts < max_attempts:
            x = random.uniform(margin, self.arena.width - margin)
            y = random.uniform(margin, self.arena.height - margin)
            if not self._is_valid_spawn(x, y, margin):
                attempts += 1
                continue
            overlap = False
            for d in self.drones:
                dx = x - d.position.x
                dy = y - d.position.y
                if dx * dx + dy * dy < (2 * margin) ** 2:
                    overlap = True
                    break
            if overlap:
                attempts += 1
                continue
            # Random initial thrust and steer for demo mode (forward only)
            thrust = random.uniform(0.5, 1.0)
            steer = random.uniform(-0.5, 0.5)
            heading = random.uniform(0, 2 * math.pi)
            vx = thrust * DRONE_SPEED * math.cos(heading)
            vy = thrust * DRONE_SPEED * math.sin(heading)
            drone = Drone(x, y, radius=DRONE_RADIUS, vx=vx, vy=vy)
            drone.thrust = thrust
            drone.steer = steer
            drone.heading = heading
            self.drones.append(drone)
            attempts = 0

    def _apply_actions(self, actions: dict[int, tuple[float, float]] | None) -> None:
        """Apply helicopter-style actions: (thrust, steer). thrust in [-1,1], steer in [-1,1]. Missing agents keep current."""
        if actions is None:
            return
        for agent_id, (thrust, steer) in actions.items():
            if 0 <= agent_id < len(self.drones):
                d = self.drones[agent_id]
                d.thrust = max(0.0, min(1.0, thrust))  # forward only
                d.steer = max(-1.0, min(1.0, steer))

    def _physics_step(self) -> None:
        """Apply movement and collision. Environment logic only; drones are passive."""
        for drone in self.drones:
            drone.update(self.dt)

        # Obstacle repulsion (soft push away, 1/distance)
        for drone in self.drones:
            repulsion = _obstacle_repulsion(drone.position, self.obstacles)
            drone.velocity += repulsion
            # Cap speed so repulsion doesn't launch drones
            if drone.velocity.length_squared() > DRONE_SPEED * DRONE_SPEED:
                drone.velocity.scale_to_length(DRONE_SPEED)

        for drone in self.drones:
            clamped = self.arena.clamp(drone.position, drone.radius)
            if clamped.x != drone.position.x or clamped.y != drone.position.y:
                drone.velocity = pygame.math.Vector2(0, 0)
            drone.position = clamped

        for drone in self.drones:
            for obs in self.obstacles:
                if _circle_rect_overlap(
                    (drone.position.x, drone.position.y),
                    drone.radius,
                    obs.get_collision_rect(),
                ):
                    new_pos, normal = _push_circle_out_of_rect(
                        drone.position, drone.radius, obs.get_collision_rect()
                    )
                    drone.position = new_pos
                    drone.velocity = pygame.math.Vector2(0, 0)

        for i in range(len(self.drones)):
            for j in range(i + 1, len(self.drones)):
                d1, d2 = self.drones[i], self.drones[j]
                (c1, r1), (c2, r2) = d1.get_collision_circle(), d2.get_collision_circle()
                if _circle_circle_overlap(c1, r1, c2, r2):
                    p1, p2 = _push_circles_apart(
                        d1.position, r1, d2.position, r2
                    )
                    d1.position = p1
                    d2.position = p2
                    d1.velocity = pygame.math.Vector2(0, 0)
                    d2.velocity = pygame.math.Vector2(0, 0)

    def _compute_observations(self) -> dict[int, dict]:
        """Compute local observations for all agents using neighbor finder."""
        positions = [(d.position.x, d.position.y) for d in self.drones]
        obstacle_rects = [obs.get_collision_rect() for obs in self.obstacles]
        perception_range = self.drones[0].perception_range if self.drones else 0

        drone_neighbors = self.neighbor_finder.find_drone_neighbors(
            positions, perception_range, exclude_self=True
        )
        obstacle_neighbors = self.neighbor_finder.find_nearby_obstacles(
            positions, obstacle_rects, perception_range
        )

        x_min, y_min, x_max, y_max = self.arena.get_bounds()
        observations: dict[int, dict] = {}
        for i, drone in enumerate(self.drones):
            px, py = drone.position.x, drone.position.y
            neighbors = []
            for j in drone_neighbors[i]:
                d = self.drones[j]
                dx = d.position.x - px
                dy = d.position.y - py
                dist = math.sqrt(dx * dx + dy * dy)
                neighbors.append({
                    "relative_pos": (dx, dy),
                    "distance": dist,
                    "velocity": (d.velocity.x, d.velocity.y),
                    "id": j,
                })
            obstacles = []
            for oi, dist, rel in obstacle_neighbors[i]:
                obstacles.append({
                    "relative_pos": rel,
                    "distance": dist,
                    "rect": obstacle_rects[oi],
                })
            observations[i] = {
                "obstacles": obstacles,
                "neighbors": neighbors,
                "boundaries": {
                    "left": px - x_min,
                    "right": x_max - px,
                    "top": py - y_min,
                    "bottom": y_max - py,
                },
                "self_state": {
                    "position": (px, py),
                    "velocity": (drone.velocity.x, drone.velocity.y),
                    "heading": drone.heading,
                },
            }
        return observations

    def step(
        self,
        actions: dict[int, tuple[float, float]] | None = None,
    ) -> tuple[
        dict[int, dict],
        dict[int, float],
        dict[int, bool],
        dict[int, bool],
        dict[str, Any],
    ]:
        """
        Step environment. Returns (observations, rewards, terminations, truncations, infos).

        actions: agent_id -> (thrust, steer). Helicopter-style:
          thrust in [-1, 1]: -1=backward, 0=no forward/back, 1=forward
          steer in [-1, 1]: -1=turn left, 0=no turn, 1=turn right
        Thrust+steer = curved path. Steer only = spin in place.
        If None, use random thrust/steer (demo mode). If agent missing, keep current.
        Compatible with PettingZoo parallel API.
        """
        if actions is None:
            # Change direction every ~0.5 s so drones wander instead of jitter
            # Thrust always has meaningful magnitude to avoid "stop and spin" (thrust≈0 + steer high)
            if self._step_count % 30 == 0:
                actions = {}
                for i in range(len(self.drones)):
                    thrust = random.uniform(0.4, 1.0)  # forward only (no backward)
                    steer = random.uniform(-0.7, 0.7)
                    actions[i] = (thrust, steer)
                self._random_actions = actions
            else:
                actions = getattr(self, "_random_actions", None)
                if actions is None:
                    actions = {
                        i: (random.uniform(0.4, 1.0), random.uniform(-0.7, 0.7))
                        for i in range(len(self.drones))
                    }
            self._step_count += 1
        self._apply_actions(actions)
        self._physics_step()
        observations = self._compute_observations()
        n = len(self.drones)
        rewards = {i: 0.0 for i in range(n)}
        terminations = {i: False for i in range(n)}
        truncations = {i: False for i in range(n)}
        infos = {"agents": list(range(n))}
        return observations, rewards, terminations, truncations, infos

    def get_observation(self, agent_id: int) -> dict | None:
        """Get local observation for one agent. Uses neighbor finder."""
        if agent_id < 0 or agent_id >= len(self.drones):
            return None
        obs = self._compute_observations()
        return obs.get(agent_id)

    def render(self, screen: pygame.Surface) -> None:
        self.arena.draw(screen)
        for obs in self.obstacles:
            obs.draw(screen)
        for drone in self.drones:
            drone.draw(screen)

    def get_drone_positions(self) -> list[tuple[float, float]]:
        return [(d.position.x, d.position.y) for d in self.drones]

    def get_drone_velocities(self) -> list[tuple[float, float]]:
        return [(d.velocity.x, d.velocity.y) for d in self.drones]

    def get_obstacles(self) -> list[pygame.Rect]:
        return [obs.get_collision_rect() for obs in self.obstacles]

    def get_neighbors(self, drone_id: int, radius: float | None = None) -> list[int]:
        if drone_id < 0 or drone_id >= len(self.drones):
            return []
        drone = self.drones[drone_id]
        r = radius if radius is not None else drone.perception_range
        pos = drone.position
        result = []
        for i, d in enumerate(self.drones):
            if i == drone_id:
                continue
            dx = d.position.x - pos.x
            dy = d.position.y - pos.y
            if dx * dx + dy * dy <= r * r:
                result.append(i)
        return result

    def get_perception(self, drone_id: int) -> dict:
        """Legacy: same as get_observation. Prefer get_observation for RL."""
        obs = self.get_observation(drone_id)
        if obs is None:
            return {"obstacles": [], "neighbors": [], "boundaries": {}}
        return {
            "obstacles": obs["obstacles"],
            "neighbors": obs["neighbors"],
            "boundaries": obs["boundaries"],
        }
