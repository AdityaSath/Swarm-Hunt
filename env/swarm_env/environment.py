"""
V1 pursuit environment core.

Integer-indexed internal API — no PettingZoo / Gymnasium dependency.
The thin PettingZoo wrapper lives in ``parallel_env.py``.

Step order:
    1. Apply predator desired velocities
    2. Scripted prey policy
    3. Physics (walls, obstacles, collisions)
    4. Capture geometry + tactical FSM
    5. Rewards
    6. Observations
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pygame

from swarm_env.arena import Arena
from swarm_env.obstacle import Obstacle
from swarm_env.drone import Drone
from swarm_env.prey import Prey
from swarm_env.capture import (
    compute_escape_gap,
    GapResult,
    TacticalFSM,
    PreyTacticalState,
    EpisodeState,
)
from swarm_env.config import (
    ARENA_WIDTH,
    ARENA_HEIGHT,
    DRONE_COUNT,
    DRONE_RADIUS,
    DRONE_SPEED,
    PREY_RADIUS,
    OBSTACLE_POSITIONS,
    DT,
    MAX_STEPS,
    R_SENSE,
    K_TEAMMATES,
    M_OBSTACLES,
    WORLD_SCALE,
    REWARD_CAPTURE,
    REWARD_TIMEOUT,
    REWARD_THREATENED,
    REWARD_CONTAINED,
    REWARD_CONTAINMENT_STEP,
    REWARD_ESCAPE,
    PENALTY_OBSTACLE_COLLISION,
    PENALTY_PREDATOR_COLLISION,
    PENALTY_IDLE,
    IDLE_SPEED_THRESHOLD,
    DIST_SHAPING_CLIP,
    CONTRIBUTOR_BONUS,
    CONTRIBUTOR_BONUS_ENABLED,
)

# Per-predator observation size:
#   self: pos(2) + vel(2) = 4
#   prey: visible(1) + rel_pos(2) + rel_vel(2) + dist(1) = 6
#   K teammates: each valid(1) + rel_pos(2) + rel_vel(2) + dist(1) = 6  → K*6
#   M obstacles: each valid(1) + rel_pos(2) + char_radius(1) + dist(1) = 5  → M*5
#   borders: left, right, top, bottom = 4
OBS_SELF = 4
OBS_PREY = 6
OBS_TEAMMATE_SLOT = 6
OBS_OBSTACLE_SLOT = 5
OBS_BORDER = 4
OBS_SIZE = OBS_SELF + OBS_PREY + K_TEAMMATES * OBS_TEAMMATE_SLOT + M_OBSTACLES * OBS_OBSTACLE_SLOT + OBS_BORDER


# ── collision helpers (pure functions) ────────────────────────────────────

def _circle_rect_overlap(cx: float, cy: float, r: float, rect: pygame.Rect) -> bool:
    closest_x = max(rect.left, min(rect.right, cx))
    closest_y = max(rect.top, min(rect.bottom, cy))
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy < r * r


def _push_circle_out_of_rect(
    center: pygame.math.Vector2, radius: float, rect: pygame.Rect
) -> tuple[pygame.math.Vector2, bool]:
    """Push circle out of rect. Returns (new_pos, collided)."""
    cx, cy = center.x, center.y
    closest_x = max(rect.left, min(rect.right, cx))
    closest_y = max(rect.top, min(rect.bottom, cy))
    dx = cx - closest_x
    dy = cy - closest_y
    dist_sq = dx * dx + dy * dy

    if dist_sq >= radius * radius:
        return center, False

    if dist_sq == 0:
        to_left = cx - rect.left
        to_right = rect.right - cx
        to_top = cy - rect.top
        to_bottom = rect.bottom - cy
        min_dist = min(to_left, to_right, to_top, to_bottom)
        if min_dist == to_left:
            return pygame.math.Vector2(rect.left - radius, cy), True
        if min_dist == to_right:
            return pygame.math.Vector2(rect.right + radius, cy), True
        if min_dist == to_top:
            return pygame.math.Vector2(cx, rect.top - radius), True
        return pygame.math.Vector2(cx, rect.bottom + radius), True

    dist = math.sqrt(dist_sq)
    overlap = radius - dist
    nx = dx / dist
    ny = dy / dist
    return center + pygame.math.Vector2(nx * overlap, ny * overlap), True


def _push_circles_apart(
    pos1: pygame.math.Vector2, r1: float,
    pos2: pygame.math.Vector2, r2: float,
) -> tuple[pygame.math.Vector2, pygame.math.Vector2, bool]:
    """Push two circles apart. Returns (new_p1, new_p2, collided)."""
    dx = pos2.x - pos1.x
    dy = pos2.y - pos1.y
    dist_sq = dx * dx + dy * dy
    combined = r1 + r2
    if dist_sq >= combined * combined:
        return pos1, pos2, False
    if dist_sq == 0:
        return (
            pos1 + pygame.math.Vector2(r1, 0),
            pos2 + pygame.math.Vector2(-r2, 0),
            True,
        )
    dist = math.sqrt(dist_sq)
    overlap = combined - dist
    nx = dx / dist
    ny = dy / dist
    half = overlap / 2
    return (
        pos1 - pygame.math.Vector2(nx * half, ny * half),
        pos2 + pygame.math.Vector2(nx * half, ny * half),
        True,
    )


# ── environment ───────────────────────────────────────────────────────────

class Environment:
    """
    V1 pursuit environment.  All public API uses **integer** agent indices.

    ``reset()`` → (observations, infos)
    ``step(actions)`` → (observations, rewards, terminations, truncations, infos)
    """

    def __init__(
        self,
        width: int = ARENA_WIDTH,
        height: int = ARENA_HEIGHT,
        drone_count: int = DRONE_COUNT,
        dt: float = DT,
        seed: int | None = None,
    ):
        self.dt = dt
        self._width = width
        self._height = height
        self._drone_count = drone_count
        self.arena = Arena(width, height)

        self.obstacles: list[Obstacle] = []
        self.drones: list[Drone] = []
        self.prey: Prey | None = None

        self._fsm = TacticalFSM()
        self._episode_state = EpisodeState.IN_PURSUIT
        self._step_count = 0

        # reward bookkeeping
        self._prev_mean_dist: float = 0.0
        self._prev_tactical: PreyTacticalState = PreyTacticalState.FREE

        # per-step collision flags (set during physics, consumed by rewards)
        self._obs_collisions: list[bool] = []
        self._pred_collisions: list[bool] = []

        # demo-mode wandering: persist random velocities across frames
        self._demo_actions: dict[int, tuple[float, float]] = {}
        self._demo_change_interval = 60  # re-pick direction every N steps (~1 s at 60 FPS)

        self.reset(seed=seed)

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
        self._step_count = 0
        self._episode_state = EpisodeState.IN_PURSUIT
        self._fsm.reset()
        self.obstacles.clear()
        self.drones.clear()
        self.prey = None
        self._init_obstacles()
        self._init_drones()
        self._init_prey()

        self._prev_mean_dist = self._mean_pred_prey_dist()
        self._prev_tactical = PreyTacticalState.FREE
        self._obs_collisions = [False] * self._drone_count
        self._pred_collisions = [False] * self._drone_count

        observations = self._compute_observations()
        infos: dict[str, Any] = {"episode_state": self._episode_state, "tactical_state": self._fsm.state}
        return observations, infos

    # ── step ──────────────────────────────────────────────────────────────

    def step(
        self,
        actions: dict[int, tuple[float, float]] | None = None,
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, float],
        dict[int, bool],
        dict[int, bool],
        dict[str, Any],
    ]:
        n = len(self.drones)
        self._step_count += 1

        # 1. apply predator desired velocities
        self._apply_actions(actions)

        # 2. scripted prey policy
        if self.prey is not None:
            pred_pos = [(d.position.x, d.position.y) for d in self.drones]
            obs_rects = [o.get_collision_rect() for o in self.obstacles]
            self.prey.decide(pred_pos, obs_rects, self._width, self._height)

        # 3. physics
        self._physics_step()

        # 4. capture / FSM
        gap, tactical = self._update_capture()

        # 5. rewards
        rewards = self._compute_rewards(gap, tactical)

        # 6. episode termination / truncation
        terminations = {i: False for i in range(n)}
        truncations = {i: False for i in range(n)}

        if tactical == PreyTacticalState.CAPTURED:
            self._episode_state = EpisodeState.CAPTURED
            for i in range(n):
                terminations[i] = True
        elif self._step_count >= MAX_STEPS:
            self._episode_state = EpisodeState.TIMEOUT
            for i in range(n):
                truncations[i] = True

        # 7. observations
        observations = self._compute_observations()

        infos: dict[str, Any] = {
            "episode_state": self._episode_state,
            "tactical_state": tactical,
            "step": self._step_count,
            "gap": gap,
        }

        self._prev_tactical = tactical

        return observations, rewards, terminations, truncations, infos

    # ── action application ────────────────────────────────────────────────

    def _apply_actions(self, actions: dict[int, tuple[float, float]] | None) -> None:
        if actions is None:
            # demo wandering: pick a new random direction every N steps,
            # otherwise keep the previous velocity so drones travel smoothly.
            if self._step_count % self._demo_change_interval == 1 or not self._demo_actions:
                self._demo_actions = {}
                for i in range(len(self.drones)):
                    angle = random.uniform(-math.pi, math.pi)
                    speed = random.uniform(0.3, 0.7) * DRONE_SPEED
                    self._demo_actions[i] = (
                        speed * math.cos(angle),
                        speed * math.sin(angle),
                    )
            for i, d in enumerate(self.drones):
                vx, vy = self._demo_actions[i]
                d.set_desired_velocity(vx, vy)
            return
        for agent_id, (vx, vy) in actions.items():
            if 0 <= agent_id < len(self.drones):
                self.drones[agent_id].set_desired_velocity(vx, vy)

    # ── physics ───────────────────────────────────────────────────────────

    def _physics_step(self) -> None:
        self._obs_collisions = [False] * len(self.drones)
        self._pred_collisions = [False] * len(self.drones)

        # integrate predators
        for drone in self.drones:
            drone.integrate(self.dt)

        # predator–wall clamp
        for drone in self.drones:
            clamped = self.arena.clamp(drone.position, drone.radius)
            if clamped.x != drone.position.x or clamped.y != drone.position.y:
                drone.velocity = pygame.math.Vector2(0, 0)
            drone.position = clamped

        # predator–obstacle hard collision
        for idx, drone in enumerate(self.drones):
            for obs in self.obstacles:
                new_pos, hit = _push_circle_out_of_rect(
                    drone.position, drone.radius, obs.get_collision_rect(),
                )
                if hit:
                    drone.position = new_pos
                    drone.velocity = pygame.math.Vector2(0, 0)
                    self._obs_collisions[idx] = True

        # predator–predator collision
        for i in range(len(self.drones)):
            for j in range(i + 1, len(self.drones)):
                d1, d2 = self.drones[i], self.drones[j]
                p1, p2, hit = _push_circles_apart(
                    d1.position, d1.radius, d2.position, d2.radius,
                )
                if hit:
                    d1.position = p1
                    d2.position = p2
                    d1.velocity = pygame.math.Vector2(0, 0)
                    d2.velocity = pygame.math.Vector2(0, 0)
                    self._pred_collisions[i] = True
                    self._pred_collisions[j] = True

        # prey integration (ignores obstacles, clip to arena only)
        if self.prey is not None:
            self.prey.integrate(self.dt)
            clamped = self.arena.clamp(self.prey.position, self.prey.radius)
            if clamped.x != self.prey.position.x or clamped.y != self.prey.position.y:
                self.prey.velocity = pygame.math.Vector2(0, 0)
            self.prey.position = clamped

    # ── capture / FSM ─────────────────────────────────────────────────────

    def _update_capture(self) -> tuple[GapResult, PreyTacticalState]:
        if self.prey is None:
            dummy = GapResult(2 * math.pi, 0.0, 0, [], 0)
            return dummy, PreyTacticalState.FREE

        px, py = self.prey.position.x, self.prey.position.y
        pred_pos = [(d.position.x, d.position.y) for d in self.drones]
        gap = compute_escape_gap(px, py, pred_pos, self._width, self._height)
        tactical = self._fsm.update(gap, pred_pos, px, py)
        return gap, tactical

    # ── rewards ───────────────────────────────────────────────────────────

    def _compute_rewards(self, gap: GapResult, tactical: PreyTacticalState) -> dict[int, float]:
        n = len(self.drones)
        shared = 0.0

        # terminal
        if tactical == PreyTacticalState.CAPTURED:
            shared += REWARD_CAPTURE
        elif self._step_count >= MAX_STEPS:
            shared += REWARD_TIMEOUT

        # tactical transitions
        prev = self._prev_tactical
        if prev == PreyTacticalState.FREE and tactical == PreyTacticalState.THREATENED:
            shared += REWARD_THREATENED
        if prev == PreyTacticalState.THREATENED and tactical == PreyTacticalState.CONTAINED:
            shared += REWARD_CONTAINED
        if (
            prev in (PreyTacticalState.CONTAINED, PreyTacticalState.THREATENED)
            and tactical == PreyTacticalState.FREE
        ):
            shared += REWARD_ESCAPE

        # containment maintenance
        if tactical == PreyTacticalState.CONTAINED:
            shared += REWARD_CONTAINMENT_STEP

        # distance shaping (clipped)
        mean_dist = self._mean_pred_prey_dist()
        delta = self._prev_mean_dist - mean_dist  # positive = got closer
        shared += max(-DIST_SHAPING_CLIP, min(DIST_SHAPING_CLIP, delta / WORLD_SCALE))
        self._prev_mean_dist = mean_dist

        # penalties (shared)
        for i in range(n):
            if self._obs_collisions[i]:
                shared += PENALTY_OBSTACLE_COLLISION / n
            if self._pred_collisions[i]:
                shared += PENALTY_PREDATOR_COLLISION / n

        # idle penalty
        for d in self.drones:
            if d.velocity.length() < IDLE_SPEED_THRESHOLD:
                shared += PENALTY_IDLE / n

        rewards = {i: shared for i in range(n)}

        # optional per-agent contributor bonus
        if CONTRIBUTOR_BONUS_ENABLED:
            for idx in gap.contributor_indices:
                if 0 <= idx < n:
                    rewards[idx] += CONTRIBUTOR_BONUS

        return rewards

    def _mean_pred_prey_dist(self) -> float:
        if self.prey is None or not self.drones:
            return 0.0
        px, py = self.prey.position.x, self.prey.position.y
        total = sum(math.hypot(d.position.x - px, d.position.y - py) for d in self.drones)
        return total / len(self.drones)

    # ── observations ──────────────────────────────────────────────────────

    def _compute_observations(self) -> dict[int, np.ndarray]:
        observations: dict[int, np.ndarray] = {}
        x_min, y_min, x_max, y_max = self.arena.get_bounds()
        obs_rects = [o.get_collision_rect() for o in self.obstacles]

        for i, drone in enumerate(self.drones):
            obs = np.zeros(OBS_SIZE, dtype=np.float32)
            offset = 0
            px, py = drone.position.x, drone.position.y

            # ---- self (4) ----
            obs[offset] = px / WORLD_SCALE
            obs[offset + 1] = py / WORLD_SCALE
            obs[offset + 2] = drone.velocity.x / DRONE_SPEED
            obs[offset + 3] = drone.velocity.y / DRONE_SPEED
            offset += OBS_SELF

            # ---- prey (6) ----
            if self.prey is not None:
                prx, pry = self.prey.position.x, self.prey.position.y
                dist_prey = math.hypot(prx - px, pry - py)
                if dist_prey <= R_SENSE:
                    obs[offset] = 1.0  # prey_visible
                    obs[offset + 1] = (prx - px) / WORLD_SCALE
                    obs[offset + 2] = (pry - py) / WORLD_SCALE
                    obs[offset + 3] = (self.prey.velocity.x - drone.velocity.x) / DRONE_SPEED
                    obs[offset + 4] = (self.prey.velocity.y - drone.velocity.y) / DRONE_SPEED
                    obs[offset + 5] = dist_prey / WORLD_SCALE
            offset += OBS_PREY

            # ---- K teammates (K * 6) ----
            teammates: list[tuple[float, int]] = []
            for j, other in enumerate(self.drones):
                if j == i:
                    continue
                dx = other.position.x - px
                dy = other.position.y - py
                d = math.hypot(dx, dy)
                if d <= R_SENSE:
                    teammates.append((d, j))
            teammates.sort()  # ascending by distance, stable on index
            for k in range(K_TEAMMATES):
                slot = offset + k * OBS_TEAMMATE_SLOT
                if k < len(teammates):
                    d, j = teammates[k]
                    other = self.drones[j]
                    obs[slot] = 1.0  # valid
                    obs[slot + 1] = (other.position.x - px) / WORLD_SCALE
                    obs[slot + 2] = (other.position.y - py) / WORLD_SCALE
                    obs[slot + 3] = (other.velocity.x - drone.velocity.x) / DRONE_SPEED
                    obs[slot + 4] = (other.velocity.y - drone.velocity.y) / DRONE_SPEED
                    obs[slot + 5] = d / WORLD_SCALE
                # else: zeros (already initialized)
            offset += K_TEAMMATES * OBS_TEAMMATE_SLOT

            # ---- M obstacles (M * 5) ----
            nearby_obs: list[tuple[float, int]] = []
            for oi, rect in enumerate(obs_rects):
                cx = max(rect.left, min(rect.right, px))
                cy = max(rect.top, min(rect.bottom, py))
                d = math.hypot(px - cx, py - cy)
                if d <= R_SENSE:
                    nearby_obs.append((d, oi))
            nearby_obs.sort()
            for m in range(M_OBSTACLES):
                slot = offset + m * OBS_OBSTACLE_SLOT
                if m < len(nearby_obs):
                    d, oi = nearby_obs[m]
                    rect = obs_rects[oi]
                    ocx, ocy = rect.centerx, rect.centery
                    char_r = math.hypot(rect.width, rect.height) / 2
                    obs[slot] = 1.0  # valid
                    obs[slot + 1] = (ocx - px) / WORLD_SCALE
                    obs[slot + 2] = (ocy - py) / WORLD_SCALE
                    obs[slot + 3] = char_r / WORLD_SCALE
                    obs[slot + 4] = d / WORLD_SCALE
            offset += M_OBSTACLES * OBS_OBSTACLE_SLOT

            # ---- borders (4) ----
            obs[offset] = (px - x_min) / WORLD_SCALE
            obs[offset + 1] = (x_max - px) / WORLD_SCALE
            obs[offset + 2] = (py - y_min) / WORLD_SCALE
            obs[offset + 3] = (y_max - py) / WORLD_SCALE

            observations[i] = obs

        return observations

    # ── spawn ─────────────────────────────────────────────────────────────

    def _init_obstacles(self) -> None:
        for x, y, size_type in OBSTACLE_POSITIONS:
            self.obstacles.append(Obstacle(x, y, size_type))

    def _init_drones(self) -> None:
        margin = DRONE_RADIUS * 2.5
        min_pair_dist = DRONE_RADIUS * 4
        max_attempts = self._drone_count * 200
        attempts = 0
        while len(self.drones) < self._drone_count and attempts < max_attempts:
            x = random.uniform(margin, self._width - margin)
            y = random.uniform(margin, self._height - margin)
            if not self._is_valid_spawn(x, y, DRONE_RADIUS):
                attempts += 1
                continue
            too_close = any(
                math.hypot(x - d.position.x, y - d.position.y) < min_pair_dist
                for d in self.drones
            )
            if too_close:
                attempts += 1
                continue
            self.drones.append(Drone(x, y, radius=DRONE_RADIUS))
            attempts = 0
        if len(self.drones) < self._drone_count:
            raise RuntimeError(
                f"Could not spawn {self._drone_count} predators without overlap "
                f"(got {len(self.drones)}). Reduce obstacle density or arena constraints."
            )

    def _init_prey(self) -> None:
        margin = PREY_RADIUS
        max_attempts = 500
        for _ in range(max_attempts):
            x = random.uniform(margin, self._width - margin)
            y = random.uniform(margin, self._height - margin)
            # prey may spawn on obstacles (it passes through them),
            # but must not start already captured
            pred_pos = [(d.position.x, d.position.y) for d in self.drones]
            gap = compute_escape_gap(x, y, pred_pos, self._width, self._height)
            if gap.largest_gap >= math.radians(120):
                self.prey = Prey(x, y, radius=PREY_RADIUS)
                return
        # fallback: spawn at arena center
        self.prey = Prey(self._width / 2, self._height / 2, radius=PREY_RADIUS)

    def _is_valid_spawn(self, x: float, y: float, radius: float) -> bool:
        if x - radius < 0 or x + radius > self._width:
            return False
        if y - radius < 0 or y + radius > self._height:
            return False
        for obs in self.obstacles:
            if _circle_rect_overlap(x, y, radius, obs.get_collision_rect()):
                return False
        return True

    # ── rendering ─────────────────────────────────────────────────────────

    def render(self, screen: pygame.Surface) -> None:
        self.arena.draw(screen)
        for obs in self.obstacles:
            obs.draw(screen)
        for drone in self.drones:
            drone.draw(screen)
        if self.prey is not None:
            self.prey.draw(screen)

    # ── accessors (used by wrapper / tests) ───────────────────────────────

    @property
    def num_agents(self) -> int:
        return len(self.drones)

    @property
    def obs_size(self) -> int:
        return OBS_SIZE
