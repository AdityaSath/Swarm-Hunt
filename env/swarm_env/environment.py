"""
V1 pursuit environment core.

Integer-indexed internal API — no PettingZoo / Gymnasium dependency.
The thin PettingZoo wrapper lives in ``parallel_env.py``.

Step order:
    1. Apply predator desired velocities
    2. Physics (walls, obstacles, collisions; prey bounces on arena edges)
    3. Capture geometry + tactical FSM
    4. Rewards
    5. Episode termination / truncation
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
    CaptureStatus,
    TacticalFSM,
    PreyTacticalState,
    EpisodeState,
    predators_in_capture_range,
)
from swarm_env.config import (
    ARENA_WIDTH,
    ARENA_HEIGHT,
    DRONE_COUNT,
    DRONE_RADIUS,
    DRONE_SPEED,
    PREY_RADIUS,
    PREY_SPEED,
    PREY_BOUNCE_SPEED_SCALE,
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
    PENALTY_OBSTACLE_COLLISION,
    PENALTY_PREDATOR_COLLISION,
    PENALTY_IDLE,
    IDLE_SPEED_THRESHOLD,
    TEAM_MEAN_PROGRESS_CLIP,
    TEAM_MEAN_PROGRESS_WEIGHT,
    TEAM_CAPTURE_RANGE_WEIGHT,
    TEAM_HOLD_PROGRESS_WEIGHT,
    REWARD_VELOCITY_TOWARD_PREY,
    VELOCITY_TOWARD_MIN_DIST,
    CHASE_BOOTSTRAP_STEPS,
    CHASE_BOOTSTRAP_MULT,
    REWARD_IN_CAPTURE_RING_PER_STEP,
    REWARD_SLOW_IN_RING,
    BOUNDARY_MARGIN_PENALTY,
    PENALTY_BOUNDARY_PROXIMITY,
    EDGE_STRAGGLER_BAND_PX,
    PENALTY_EDGE_STRAGGLER,
    STRAGGLER_DIST_SCALE,
    STUCK_EDGE_MARGIN,
    STUCK_SPEED_THRESHOLD,
    STUCK_STEPS,
    PENALTY_STUCK,
    WALL_TANGENT_DAMPING,
    OBSTACLE_TANGENT_DAMPING,
    PREDATOR_TANGENT_DAMPING,
    R_CAPTURE_RANGE,
    R_DANGER,
    CONTRIBUTOR_BONUS,
    CONTRIBUTOR_BONUS_ENABLED,
    CAPTURE_HOLD_STEPS,
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
) -> tuple[pygame.math.Vector2, bool, pygame.math.Vector2 | None]:
    """Push circle out of rect. Returns (new_pos, collided, outward_normal)."""
    cx, cy = center.x, center.y
    closest_x = max(rect.left, min(rect.right, cx))
    closest_y = max(rect.top, min(rect.bottom, cy))
    dx = cx - closest_x
    dy = cy - closest_y
    dist_sq = dx * dx + dy * dy

    if dist_sq >= radius * radius:
        return center, False, None

    if dist_sq == 0:
        to_left = cx - rect.left
        to_right = rect.right - cx
        to_top = cy - rect.top
        to_bottom = rect.bottom - cy
        min_dist = min(to_left, to_right, to_top, to_bottom)
        if min_dist == to_left:
            return pygame.math.Vector2(rect.left - radius, cy), True, pygame.math.Vector2(-1, 0)
        if min_dist == to_right:
            return pygame.math.Vector2(rect.right + radius, cy), True, pygame.math.Vector2(1, 0)
        if min_dist == to_top:
            return pygame.math.Vector2(cx, rect.top - radius), True, pygame.math.Vector2(0, -1)
        return pygame.math.Vector2(cx, rect.bottom + radius), True, pygame.math.Vector2(0, 1)

    dist = math.sqrt(dist_sq)
    overlap = radius - dist
    nx = dx / dist
    ny = dy / dist
    normal = pygame.math.Vector2(nx, ny)
    return center + pygame.math.Vector2(nx * overlap, ny * overlap), True, normal


def _push_circles_apart(
    pos1: pygame.math.Vector2, r1: float,
    pos2: pygame.math.Vector2, r2: float,
) -> tuple[
    pygame.math.Vector2,
    pygame.math.Vector2,
    bool,
    pygame.math.Vector2 | None,
    pygame.math.Vector2 | None,
]:
    """Push two circles apart. Returns (new_p1, new_p2, collided, n1, n2)."""
    dx = pos2.x - pos1.x
    dy = pos2.y - pos1.y
    dist_sq = dx * dx + dy * dy
    combined = r1 + r2
    if dist_sq >= combined * combined:
        return pos1, pos2, False, None, None
    if dist_sq == 0:
        return (
            pos1 + pygame.math.Vector2(r1, 0),
            pos2 + pygame.math.Vector2(-r2, 0),
            True,
            pygame.math.Vector2(1, 0),
            pygame.math.Vector2(-1, 0),
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
        pygame.math.Vector2(-nx, -ny),
        pygame.math.Vector2(nx, ny),
    )


def _slide_velocity(
    velocity: pygame.math.Vector2,
    normal: pygame.math.Vector2 | None,
    tangent_damping: float = 1.0,
) -> pygame.math.Vector2:
    """Remove inward motion while preserving tangential slide along the contact surface."""
    if normal is None:
        return velocity
    n = pygame.math.Vector2(normal)
    if n.length_squared() == 0:
        return velocity
    n = n.normalize()
    inward = velocity.dot(n)
    if inward >= 0.0:
        return velocity
    tangential = velocity - n * inward
    return tangential * tangent_damping


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
        prey_speed_factor: float = 1.0,
        prey_bounce_speed_scale: float | None = None,
    ):
        self.dt = dt
        self._width = width
        self._height = height
        self._drone_count = drone_count
        self._prey_speed_factor = max(0.0, prey_speed_factor)
        self._prey_bounce_speed_scale = prey_bounce_speed_scale
        self.arena = Arena(width, height)

        self.obstacles: list[Obstacle] = []
        self.drones: list[Drone] = []
        self.prey: Prey | None = None

        self._fsm = TacticalFSM()
        self._episode_state = EpisodeState.IN_PURSUIT
        self._step_count = 0

        # reward bookkeeping
        self._prev_indiv_dists: list[float] = []
        self._prev_mean_prey_dist = 0.0
        self._prev_tactical: PreyTacticalState = PreyTacticalState.FREE

        # per-step collision flags (set during physics, consumed by rewards)
        self._obs_collisions: list[bool] = []
        self._pred_collisions: list[bool] = []
        self._wall_contacts: list[bool] = []
        self._edge_stuck_counts: list[int] = []

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

        self._prev_indiv_dists = self._per_drone_prey_dists()
        self._prev_mean_prey_dist = (
            float(np.mean(np.asarray(self._prev_indiv_dists, dtype=np.float64)))
            if self._prev_indiv_dists
            else 0.0
        )
        self._prev_tactical = PreyTacticalState.FREE
        self._obs_collisions = [False] * self._drone_count
        self._pred_collisions = [False] * self._drone_count
        self._wall_contacts = [False] * self._drone_count
        self._edge_stuck_counts = [0] * self._drone_count

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

        # 2. physics (prey: integrate + arena bounce in _physics_step)
        self._physics_step()

        # 3. capture / FSM
        capture, tactical = self._update_capture()

        # 4. rewards
        rewards = self._compute_rewards(capture, tactical)

        # 5. episode termination / truncation
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

        # 6. observations
        observations = self._compute_observations()

        infos: dict[str, Any] = {
            "episode_state": self._episode_state,
            "tactical_state": tactical,
            "step": self._step_count,
            "capture": capture,
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
        self._wall_contacts = [False] * len(self.drones)

        # integrate predators
        for drone in self.drones:
            drone.integrate(self.dt)

        # predator–wall clamp
        for idx, drone in enumerate(self.drones):
            original = drone.position.copy()
            clamped = self.arena.clamp(drone.position, drone.radius)
            if clamped.x != original.x:
                normal_x = 1.0 if clamped.x > original.x else -1.0
                drone.velocity = _slide_velocity(
                    drone.velocity,
                    pygame.math.Vector2(normal_x, 0.0),
                    WALL_TANGENT_DAMPING,
                )
                self._wall_contacts[idx] = True
            if clamped.y != original.y:
                normal_y = 1.0 if clamped.y > original.y else -1.0
                drone.velocity = _slide_velocity(
                    drone.velocity,
                    pygame.math.Vector2(0.0, normal_y),
                    WALL_TANGENT_DAMPING,
                )
                self._wall_contacts[idx] = True
            drone.position = clamped

        # predator–obstacle hard collision
        for idx, drone in enumerate(self.drones):
            for obs in self.obstacles:
                new_pos, hit, normal = _push_circle_out_of_rect(
                    drone.position, drone.radius, obs.get_collision_rect(),
                )
                if hit:
                    drone.position = new_pos
                    drone.velocity = _slide_velocity(
                        drone.velocity,
                        normal,
                        OBSTACLE_TANGENT_DAMPING,
                    )
                    self._obs_collisions[idx] = True

        # predator–predator collision
        for i in range(len(self.drones)):
            for j in range(i + 1, len(self.drones)):
                d1, d2 = self.drones[i], self.drones[j]
                p1, p2, hit, n1, n2 = _push_circles_apart(
                    d1.position, d1.radius, d2.position, d2.radius,
                )
                if hit:
                    d1.position = p1
                    d2.position = p2
                    d1.velocity = _slide_velocity(
                        d1.velocity,
                        n1,
                        PREDATOR_TANGENT_DAMPING,
                    )
                    d2.velocity = _slide_velocity(
                        d2.velocity,
                        n2,
                        PREDATOR_TANGENT_DAMPING,
                    )
                    self._pred_collisions[i] = True
                    self._pred_collisions[j] = True

        # prey: bounce on arena edges
        if self.prey is not None:
            self.prey.integrate(self.dt)
            clamped, vel = self.arena.clamp_and_bounce(
                self.prey.position, self.prey.velocity, self.prey.radius
            )
            self.prey.position = clamped
            self.prey.velocity = vel

    # ── capture / FSM ─────────────────────────────────────────────────────

    def _update_capture(self) -> tuple[CaptureStatus, PreyTacticalState]:
        if self.prey is None:
            dummy = CaptureStatus(0, [], 0, 0)
            return dummy, PreyTacticalState.FREE

        px, py = self.prey.position.x, self.prey.position.y
        pred_pos = [(d.position.x, d.position.y) for d in self.drones]
        tactical = self._fsm.update(
            pred_pos, px, py, float(self._width), float(self._height),
        )
        n_in, indices = predators_in_capture_range(px, py, pred_pos, R_CAPTURE_RANGE)
        w = self._fsm.last_wall_count
        status = CaptureStatus(n_in, indices, self._fsm.hold_counter, w)
        return status, tactical

    # ── rewards ───────────────────────────────────────────────────────────

    def _compute_rewards(
        self, capture: CaptureStatus, tactical: PreyTacticalState
    ) -> dict[int, float]:
        n = len(self.drones)
        rewards = {i: 0.0 for i in range(n)}
        cur_dists = self._per_drone_prey_dists()
        mean_dist = (
            float(np.mean(np.asarray(cur_dists, dtype=np.float64)))
            if cur_dists
            else 0.0
        )
        team_reward = 0.0

        if tactical == PreyTacticalState.CAPTURED:
            team_reward += REWARD_CAPTURE
        elif self._step_count >= MAX_STEPS:
            team_reward += REWARD_TIMEOUT

        if (
            self._prev_tactical == PreyTacticalState.FREE
            and tactical == PreyTacticalState.THREATENED
        ):
            team_reward += REWARD_THREATENED

        delta_mean = self._prev_mean_prey_dist - mean_dist
        delta_norm = max(
            -TEAM_MEAN_PROGRESS_CLIP,
            min(TEAM_MEAN_PROGRESS_CLIP, delta_mean / WORLD_SCALE),
        )
        team_reward += TEAM_MEAN_PROGRESS_WEIGHT * delta_norm
        team_reward += TEAM_CAPTURE_RANGE_WEIGHT * (capture.in_range_count / max(1, n))
        if capture.hold_counter > 0:
            team_reward += TEAM_HOLD_PROGRESS_WEIGHT * (
                capture.hold_counter / max(1, CAPTURE_HOLD_STEPS)
            )

        for i in range(n):
            rewards[i] = team_reward

        self._prev_indiv_dists = cur_dists
        self._prev_mean_prey_dist = mean_dist

        if (
            PENALTY_EDGE_STRAGGLER > 0.0
            and EDGE_STRAGGLER_BAND_PX > 0.0
            and STRAGGLER_DIST_SCALE > 0.0
            and self.prey is not None
            and n >= 2
        ):
            median_d = float(np.median(np.asarray(cur_dists, dtype=np.float64)))
            inv_s = 1.0 / STRAGGLER_DIST_SCALE
            x_min, y_min, x_max, y_max = self.arena.get_bounds()
            for i, d in enumerate(self.drones):
                excess = cur_dists[i] - median_d
                s = 0.0 if excess <= 0.0 else min(1.0, excess * inv_s)
                px, py = d.position.x, d.position.y
                edge_dist = min(px - x_min, x_max - px, py - y_min, y_max - py)
                if edge_dist >= EDGE_STRAGGLER_BAND_PX:
                    e = 0.0
                else:
                    e = 1.0 - (edge_dist / EDGE_STRAGGLER_BAND_PX)
                    e = max(0.0, min(1.0, e))
                rewards[i] -= PENALTY_EDGE_STRAGGLER * e * s

        chase_w = (
            CHASE_BOOTSTRAP_MULT
            if self._step_count < CHASE_BOOTSTRAP_STEPS
            else 1.0
        )
        x_min, y_min, x_max, y_max = self.arena.get_bounds()
        if self.prey is not None:
            prx = self.prey.position.x
            pry = self.prey.position.y
            for i, d in enumerate(self.drones):
                dx = prx - d.position.x
                dy = pry - d.position.y
                dist = math.hypot(dx, dy)
                vx, vy = d.velocity.x, d.velocity.y
                spd = math.hypot(vx, vy)

                if dist <= R_CAPTURE_RANGE:
                    rewards[i] += REWARD_IN_CAPTURE_RING_PER_STEP
                    rewards[i] += REWARD_SLOW_IN_RING * (
                        1.0 - min(1.0, spd / DRONE_SPEED)
                    )

                if dist > VELOCITY_TOWARD_MIN_DIST:
                    ux, uy = dx / dist, dy / dist
                    toward = max(-1.0, min(1.0, (vx * ux + vy * uy) / DRONE_SPEED))
                    rewards[i] += chase_w * REWARD_VELOCITY_TOWARD_PREY * toward

                edge = min(
                    d.position.x - x_min,
                    x_max - d.position.x,
                    d.position.y - y_min,
                    y_max - d.position.y,
                )
                if edge < BOUNDARY_MARGIN_PENALTY and BOUNDARY_MARGIN_PENALTY > 0:
                    t = 1.0 - (edge / BOUNDARY_MARGIN_PENALTY)
                    rewards[i] += PENALTY_BOUNDARY_PROXIMITY * max(0.0, min(1.0, t))

                if edge <= STUCK_EDGE_MARGIN and spd <= STUCK_SPEED_THRESHOLD:
                    self._edge_stuck_counts[i] += 1
                else:
                    self._edge_stuck_counts[i] = 0
                if self._edge_stuck_counts[i] >= STUCK_STEPS:
                    stuck_scale = min(
                        1.0,
                        self._edge_stuck_counts[i] / max(1, STUCK_STEPS * 2),
                    )
                    rewards[i] += PENALTY_STUCK * stuck_scale

        for i in range(n):
            if self._obs_collisions[i]:
                rewards[i] += PENALTY_OBSTACLE_COLLISION
            if self._pred_collisions[i]:
                rewards[i] += PENALTY_PREDATOR_COLLISION
            if (
                self.drones[i].velocity.length() < IDLE_SPEED_THRESHOLD
                and (i >= len(cur_dists) or cur_dists[i] > R_CAPTURE_RANGE)
            ):
                rewards[i] += PENALTY_IDLE

        if CONTRIBUTOR_BONUS_ENABLED:
            for idx in capture.contributor_indices:
                if 0 <= idx < n:
                    rewards[idx] += CONTRIBUTOR_BONUS

        return rewards

    def _per_drone_prey_dists(self) -> list[float]:
        if self.prey is None or not self.drones:
            return [0.0] * len(self.drones)
        px, py = self.prey.position.x, self.prey.position.y
        return [math.hypot(d.position.x - px, d.position.y - py) for d in self.drones]

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
                obs[offset] = 1.0
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
        scale = (
            self._prey_bounce_speed_scale
            if self._prey_bounce_speed_scale is not None
            else PREY_BOUNCE_SPEED_SCALE
        )
        prey_speed = PREY_SPEED * self._prey_speed_factor * scale
        x, y = self._spawn_prey_near_arena_center()
        ang = random.uniform(-math.pi, math.pi)
        self.prey = Prey(
            x,
            y,
            radius=PREY_RADIUS,
            speed=prey_speed,
            vx=prey_speed * math.cos(ang),
            vy=prey_speed * math.sin(ang),
        )

    def _spawn_prey_near_arena_center(self) -> tuple[float, float]:
        """Spawn near arena center on open ground (center may intersect an obstacle)."""
        cx = self._width / 2
        cy = self._height / 2
        r = PREY_RADIUS
        for ring in range(0, 361, 30):
            if ring == 0:
                candidates = [(cx, cy)]
            else:
                n = max(8, ring // 15)
                candidates = [
                    (
                        cx + ring * math.cos(2 * math.pi * i / n),
                        cy + ring * math.sin(2 * math.pi * i / n),
                    )
                    for i in range(n)
                ]
            for x, y in candidates:
                if self._is_valid_spawn(x, y, r):
                    return x, y
        for _ in range(600):
            x = random.uniform(r, self._width - r)
            y = random.uniform(r, self._height - r)
            if self._is_valid_spawn(x, y, r):
                return x, y
        raise RuntimeError("Could not find valid bounce-prey spawn (try fewer obstacles).")

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
