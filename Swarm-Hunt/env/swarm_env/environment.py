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
    R_CAP,
    MIN_PREDATOR_CONTRIBUTORS,
    T_HOLD,
    PURSUIT_WEIGHT,
    FLANK_WEIGHT,
    INERTIA_WEIGHT,
    FLANK_RADIUS_MULT,
    ENABLE_DYNAMIC_FLANK,
    FLANK_ANGLE_CANDIDATES,
    GRID_ROWS,
    GRID_COLS,
    GRID_TARGET_JITTER,
    GRID_SEARCH_SPEED_FRAC,
    GRID_CONVERGE_SPEED_FRAC,
    GRID_TARGET_REACHED_DIST,
    PURSUIT_REWARD_COEF,
    AGENT_DIST_SHAPING_CLIP,
    FLANK_REWARD,
    GRID_DISCOVERY_REWARD,
    GRID_COVERAGE_REWARD,
    PREY_GRID_DISCOVERY_REWARD,
    GRID_CONVERGE_REWARD_COEF,
    GRID_CONVERGE_SHAPING_CLIP,
    FLANK_DIVERSITY_REWARD,
    DISPERSION_PHASE_STEPS,
    DISPERSION_REWARD_COEF,
    DISPERSION_SHAPING_CLIP,
    INITIAL_SEPARATION_REWARD_COEF,
    INITIAL_SEPARATION_TARGET_DIST,
    AWAY_FROM_TEAMMATE_REWARD_COEF,
    INITIAL_UNIQUE_GRID_REWARD,
    INITIAL_OUTER_GRID_REWARD,
    INITIAL_CENTER_GRID_PENALTY,
    SEE_PREY_REWARD,
    SEE_PREY_TEAM_BONUS,
    SEE_PREY_TEAM_FRAC,
    CAP_BLOCK_DIST,
    CAP_BLOCK_ANGLE_SAMPLES,
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

        # alternate capture hold counter (distance+contributors heuristic)
        self._alt_capture_hold = 0

        # reward bookkeeping
        self._prev_mean_dist = 0.0
        self._prev_tactical = PreyTacticalState.FREE
        # per-agent distance bookkeeping for shaped pursuit rewards
        self._prev_agent_dists: list[float] = []
        # per-agent nearest-neighbor distances for dispersion shaping
        self._prev_agent_nndists: list[float] = []
        self._prev_agent_grid_target_dists: list[float] = []
        self._visited_grid_cells: set[tuple[int, int]] = set()
        self._known_prey_grid_cell: tuple[int, int] | None = None
        self._grid_search_targets: list[pygame.math.Vector2] = []
        self._last_grid_targets: list[pygame.math.Vector2] = []

        # per-step collision flags (set during physics, consumed by rewards)
        self._obs_collisions: list[bool] = []
        self._pred_collisions: list[bool] = []

        # demo-mode wandering: persist random velocities across frames
        self._demo_actions: dict[int, tuple[float, float]] = {}
        self._demo_change_interval = 60  # re-pick direction every N steps (~1 s at 60 FPS)
        # last computed flank target positions (for visualization)
        self._last_flank_targets = []

        self.reset(seed=seed)

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
        self._step_count = 0
        self._episode_state = EpisodeState.IN_PURSUIT
        self._alt_capture_hold = 0
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
        # initialize per-agent previous distances
        px, py = self.prey.position.x, self.prey.position.y
        self._prev_agent_dists = [math.hypot(d.position.x - px, d.position.y - py) for d in self.drones]
        # initialize per-agent nearest-neighbor distances
        self._prev_agent_nndists = []
        for i, d in enumerate(self.drones):
            # compute distance to nearest other predator
            nnd = min(
                (math.hypot(d.position.x - d2.position.x, d.position.y - d2.position.y) for j, d2 in enumerate(self.drones) if j != i),
                default=0.0,
            )
            self._prev_agent_nndists.append(nnd)
        self._visited_grid_cells = set()
        self._known_prey_grid_cell = None
        self._init_grid_search_targets()
        self._prev_agent_grid_target_dists = [
            d.position.distance_to(self._grid_search_targets[i])
            for i, d in enumerate(self.drones)
        ]

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

        # Keep the distance+contributors capture heuristic on the same path
        # as the angular and blocked-move capture checks, before rewards and
        # termination maps are computed.
        tactical = self._apply_alternate_capture(gap, tactical)

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
            # When no external actions are provided, drive predators with a
            # simple scripted policy: trend toward the prey while preserving
            # some inertia so motion looks natural. If no prey is available,
            # fallback to the original random demo wandering.
            if self.prey is not None:
                self._demo_actions = self.scripted_actions()
                for i, d in enumerate(self.drones):
                    vx, vy = self._demo_actions[i]
                    d.set_desired_velocity(vx, vy)
                return

            # fallback: original demo wandering when no prey exists
            if self._step_count % self._demo_change_interval == 1 or not self._demo_actions:
                self._demo_actions = {}
                for i in range(len(self.drones)):
                    angle = random.uniform(-math.pi, math.pi)
                    speed = random.uniform(0.3, 0.7) * DRONE_SPEED
                    self._demo_actions[i] = (
                        speed * math.cos(angle),
                        speed * math.sin(angle),
                    )
            # clear last flank targets when wandering
            self._last_flank_targets = []
            for i, d in enumerate(self.drones):
                vx, vy = self._demo_actions[i]
                d.set_desired_velocity(vx, vy)
            return
        for agent_id, (vx, vy) in actions.items():
            if 0 <= agent_id < len(self.drones):
                self.drones[agent_id].set_desired_velocity(vx, vy)

    # Public helper: compute the pursuit/flank scripted policy for each drone
    # without mutating velocities. Useful for demos, eval, and warm starts.
    def scripted_actions(self) -> dict[int, tuple[float, float]]:
        actions: dict[int, tuple[float, float]] = {}
        if self.prey is None:
            return actions

        n = len(self.drones)
        prey_cell = self._grid_cell_for_pos(self.prey.position.x, self.prey.position.y)
        same_cell_indices = [
            i for i, d in enumerate(self.drones)
            if self._grid_cell_for_pos(d.position.x, d.position.y) == prey_cell
        ]
        if same_cell_indices:
            self._known_prey_grid_cell = prey_cell

        if self._known_prey_grid_cell is not None:
            return self._scripted_converge_actions(prey_cell)

        return self._scripted_grid_search_actions()

    def _scripted_converge_actions(self, prey_cell: tuple[int, int]) -> dict[int, tuple[float, float]]:
        actions: dict[int, tuple[float, float]] = {}
        if self.prey is None:
            return actions

        n = len(self.drones)
        flank_radius = R_CAP * FLANK_RADIUS_MULT
        assignments = self._scripted_flank_assignments()
        targets: list[pygame.math.Vector2] = []

        mean_dist = self._mean_pred_prey_dist()
        pursuit_bias_thresh = R_CAP * 1.2
        if mean_dist > pursuit_bias_thresh:
            local_pursuit_w = 0.85
            local_flank_w = 0.15
        else:
            local_pursuit_w = PURSUIT_WEIGHT
            local_flank_w = FLANK_WEIGHT

        for i, d in enumerate(self.drones):
            pursuit = pygame.math.Vector2(
                self.prey.position.x - d.position.x,
                self.prey.position.y - d.position.y,
            )
            target_ang = assignments[i] if i < len(assignments) else 2 * math.pi * float(i) / max(1, n)
            target = pygame.math.Vector2(
                self.prey.position.x + math.cos(target_ang) * flank_radius,
                self.prey.position.y + math.sin(target_ang) * flank_radius,
            )
            flank = target - d.position

            if pursuit.length_squared() > 1e-6:
                pursuit.scale_to_length(DRONE_SPEED)
            else:
                pursuit = pygame.math.Vector2(0, 0)

            if flank.length_squared() > 1e-6:
                flank.scale_to_length(DRONE_SPEED)
            else:
                flank = pygame.math.Vector2(0, 0)

            cell_center = self._grid_center(prey_cell)
            grid_pull = cell_center - d.position
            if grid_pull.length_squared() > 1e-6:
                grid_pull.scale_to_length(DRONE_SPEED * GRID_CONVERGE_SPEED_FRAC)
            else:
                grid_pull = pygame.math.Vector2(0, 0)

            desired = (
                pursuit * local_pursuit_w
                + flank * local_flank_w
                + grid_pull * 0.1
                + d.velocity * INERTIA_WEIGHT
            )
            if desired.length_squared() > DRONE_SPEED * DRONE_SPEED:
                desired.scale_to_length(DRONE_SPEED)
            actions[i] = (desired.x, desired.y)
            targets.append(target)

        self._last_flank_targets = targets
        self._last_grid_targets = [self._grid_center(prey_cell) for _ in self.drones]
        return actions

    def _scripted_grid_search_actions(self) -> dict[int, tuple[float, float]]:
        actions: dict[int, tuple[float, float]] = {}
        if len(self._grid_search_targets) != len(self.drones):
            self._init_grid_search_targets()

        targets: list[pygame.math.Vector2] = []
        occupied = self._grid_cells_occupied_by_drones()
        for i, d in enumerate(self.drones):
            target = self._grid_search_targets[i]
            if d.position.distance_to(target) <= GRID_TARGET_REACHED_DIST:
                undercovered = [
                    (r, c)
                    for r in range(GRID_ROWS)
                    for c in range(GRID_COLS)
                    if occupied.get((r, c), 0) == 0
                ]
                cell = random.choice(undercovered) if undercovered else self._random_grid_cell()
                target = self._random_point_in_cell(cell)
                self._grid_search_targets[i] = target

            desired = target - d.position
            if desired.length_squared() > 1e-6:
                desired.scale_to_length(DRONE_SPEED * GRID_SEARCH_SPEED_FRAC)
            else:
                desired = pygame.math.Vector2(0, 0)

            separation = pygame.math.Vector2(0, 0)
            for j, other in enumerate(self.drones):
                if i == j:
                    continue
                delta = d.position - other.position
                dist_sq = delta.length_squared()
                if 1e-6 < dist_sq < (DRONE_RADIUS * 8) ** 2:
                    separation += delta.normalize() * (DRONE_SPEED / max(1.0, math.sqrt(dist_sq)))
            desired += separation * 0.35 + d.velocity * INERTIA_WEIGHT
            if desired.length_squared() > DRONE_SPEED * DRONE_SPEED:
                desired.scale_to_length(DRONE_SPEED)

            actions[i] = (desired.x, desired.y)
            targets.append(target)

        self._last_grid_targets = targets
        self._last_flank_targets = []
        return actions

    def hybrid_actions(self, intent: np.ndarray) -> dict[int, tuple[float, float]]:
        """Convert learned high-level intent into hard-clipped drone velocities.

        Per agent intent is three values in [-1, 1]:
        x/y choose a search target in the arena before prey-grid discovery;
        flank chooses the angular approach direction after discovery.
        """
        actions: dict[int, tuple[float, float]] = {}
        if self.prey is None:
            return actions

        intent = np.asarray(intent, dtype=np.float32).reshape((len(self.drones), 3))
        prey_cell = self._grid_cell_for_pos(self.prey.position.x, self.prey.position.y)
        if any(self._grid_cell_for_pos(d.position.x, d.position.y) == prey_cell for d in self.drones):
            self._known_prey_grid_cell = prey_cell

        targets: list[pygame.math.Vector2] = []
        if self._known_prey_grid_cell is None:
            for i, d in enumerate(self.drones):
                tx = (float(intent[i, 0]) + 1.0) * 0.5 * self._width
                ty = (float(intent[i, 1]) + 1.0) * 0.5 * self._height
                target = pygame.math.Vector2(tx, ty)
                desired = target - d.position
                if desired.length_squared() > 1e-6:
                    desired.scale_to_length(DRONE_SPEED * GRID_SEARCH_SPEED_FRAC)
                else:
                    desired = pygame.math.Vector2(0, 0)
                actions[i] = (desired.x, desired.y)
                targets.append(target)
            self._last_grid_targets = targets
            self._last_flank_targets = []
            return actions

        cell_center = self._grid_center(self._known_prey_grid_cell)
        flank_radius = R_CAP * FLANK_RADIUS_MULT
        flank_targets: list[pygame.math.Vector2] = []
        grid_targets: list[pygame.math.Vector2] = []
        for i, d in enumerate(self.drones):
            flank_angle = float(intent[i, 2]) * math.pi
            target = pygame.math.Vector2(
                self.prey.position.x + math.cos(flank_angle) * flank_radius,
                self.prey.position.y + math.sin(flank_angle) * flank_radius,
            )
            flank = target - d.position
            pursuit = pygame.math.Vector2(self.prey.position.x - d.position.x, self.prey.position.y - d.position.y)
            grid_pull = cell_center - d.position

            for vec, speed in ((flank, DRONE_SPEED), (pursuit, DRONE_SPEED), (grid_pull, DRONE_SPEED * GRID_CONVERGE_SPEED_FRAC)):
                if vec.length_squared() > 1e-6:
                    vec.scale_to_length(speed)

            desired = pursuit * 0.45 + flank * 0.45 + grid_pull * 0.1 + d.velocity * INERTIA_WEIGHT
            if desired.length_squared() > DRONE_SPEED * DRONE_SPEED:
                desired.scale_to_length(DRONE_SPEED)
            actions[i] = (desired.x, desired.y)
            flank_targets.append(target)
            grid_targets.append(cell_center)

        self._last_flank_targets = flank_targets
        self._last_grid_targets = grid_targets
        return actions

    def _scripted_flank_assignments(self) -> list[float]:
        n = len(self.drones)
        if self.prey is None or n == 0:
            return []
        if not ENABLE_DYNAMIC_FLANK:
            return [2 * math.pi * float(i) / n for i in range(n)]

        def _norm(a: float) -> float:
            return a % (2 * math.pi)

        def _ang_dist(a: float, b: float) -> float:
            return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

        current_angles = [
            _norm(math.atan2(d.position.y - self.prey.position.y, d.position.x - self.prey.position.x))
            for d in self.drones
        ]
        candidate_count = max(12, int(FLANK_ANGLE_CANDIDATES))
        candidates = [2 * math.pi * i / candidate_count for i in range(candidate_count)]
        candidate_scores = [
            (min(_ang_dist(candidate, angle) for angle in current_angles), candidate)
            for candidate in candidates
        ]
        candidate_scores.sort(reverse=True)

        remaining_angles = [angle for _, angle in candidate_scores[:n]]
        assignments: list[float] = [0.0] * n
        for i, d in enumerate(self.drones):
            bearing = math.atan2(d.position.y - self.prey.position.y, d.position.x - self.prey.position.x)
            if remaining_angles:
                chosen = min(remaining_angles, key=lambda angle: _ang_dist(bearing, angle))
                remaining_angles.remove(chosen)
                assignments[i] = chosen
            else:
                assignments[i] = 2 * math.pi * float(i) / n
        return assignments

    def _init_grid_search_targets(self) -> None:
        cells = [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]
        random.shuffle(cells)
        self._grid_search_targets = [
            self._random_point_in_cell(cells[i % len(cells)])
            for i in range(len(self.drones))
        ]

    def _random_grid_cell(self) -> tuple[int, int]:
        return (random.randrange(GRID_ROWS), random.randrange(GRID_COLS))

    def _grid_cell_for_pos(self, x: float, y: float) -> tuple[int, int]:
        col = min(GRID_COLS - 1, max(0, int(x / max(1.0, self._width / GRID_COLS))))
        row = min(GRID_ROWS - 1, max(0, int(y / max(1.0, self._height / GRID_ROWS))))
        return (row, col)

    def _grid_center(self, cell: tuple[int, int]) -> pygame.math.Vector2:
        row, col = cell
        cell_w = self._width / GRID_COLS
        cell_h = self._height / GRID_ROWS
        return pygame.math.Vector2((col + 0.5) * cell_w, (row + 0.5) * cell_h)

    def _random_point_in_cell(self, cell: tuple[int, int]) -> pygame.math.Vector2:
        center = self._grid_center(cell)
        cell_w = self._width / GRID_COLS
        cell_h = self._height / GRID_ROWS
        jitter_x = random.uniform(-0.5, 0.5) * cell_w * GRID_TARGET_JITTER
        jitter_y = random.uniform(-0.5, 0.5) * cell_h * GRID_TARGET_JITTER
        return pygame.math.Vector2(center.x + jitter_x, center.y + jitter_y)

    def _grid_cells_occupied_by_drones(self) -> dict[tuple[int, int], int]:
        occupied: dict[tuple[int, int], int] = {}
        for d in self.drones:
            cell = self._grid_cell_for_pos(d.position.x, d.position.y)
            occupied[cell] = occupied.get(cell, 0) + 1
        return occupied

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
        # Additional capture test: if the prey cannot move by a short
        # radius in any sampled direction without colliding with a
        # predator, treat that as capture (hard block). This captures
        # the intuitive "prey is surrounded and cannot step anywhere"
        # case even if angular gap heuristics are noisy.
        try:
            blocked = self._is_prey_blocked(CAP_BLOCK_DIST, CAP_BLOCK_ANGLE_SAMPLES)
            if blocked:
                tactical = PreyTacticalState.CAPTURED
        except Exception:
            # be conservative: if the helper fails for any reason, do not
            # change the tactical state
            pass

        return gap, tactical

    def _is_prey_blocked(self, block_dist: float, angle_samples: int) -> bool:
        """Return True if the prey cannot move by block_dist in any sampled
        direction without intersecting a predator (using predator radii).
        """
        if self.prey is None or not self.drones:
            return False
        px, py = self.prey.position.x, self.prey.position.y
        # sample angles uniformly
        for k in range(angle_samples):
            ang = 2 * math.pi * k / max(1, angle_samples)
            tx = px + math.cos(ang) * block_dist
            ty = py + math.sin(ang) * block_dist
            # check collision with any predator (circle overlap)
            free = True
            for d in self.drones:
                dist = math.hypot(d.position.x - tx, d.position.y - ty)
                if dist <= d.radius + 1e-6:
                    free = False
                    break
            if free:
                # found at least one free move
                return False
        # no free sampled direction -> blocked
        return True

    def _apply_alternate_capture(
        self,
        gap: GapResult,
        tactical: PreyTacticalState,
    ) -> PreyTacticalState:
        if (
            gap.predator_contributors >= MIN_PREDATOR_CONTRIBUTORS
            and self._mean_pred_prey_dist() <= R_CAP * 0.8
        ):
            self._alt_capture_hold += 1
            if self._alt_capture_hold >= T_HOLD:
                return PreyTacticalState.CAPTURED
        else:
            self._alt_capture_hold = 0
        return tactical

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

        # per-agent pursuit shaping: reward agents for decreasing their
        # individual distance to the prey (clipped). This encourages each
        # agent to actively close on the prey instead of idling or circling.
        if self.prey is not None and PURSUIT_REWARD_COEF != 0.0:
            px, py = self.prey.position.x, self.prey.position.y
            current_dists = [math.hypot(d.position.x - px, d.position.y - py) for d in self.drones]
            for i, d in enumerate(self.drones):
                prev = self._prev_agent_dists[i] if i < len(self._prev_agent_dists) else current_dists[i]
                delta = prev - current_dists[i]  # positive means got closer
                # normalize by world scale and clip
                shaped = max(-AGENT_DIST_SHAPING_CLIP, min(AGENT_DIST_SHAPING_CLIP, delta / WORLD_SCALE))
                rewards[i] += PURSUIT_REWARD_COEF * shaped
            # update stored distances for next step
            self._prev_agent_dists = current_dists

        # Dispersion shaping: during the initial phase of the episode,
        # reward agents for increasing and maintaining teammate separation.
        if self._step_count <= DISPERSION_PHASE_STEPS and DISPERSION_REWARD_COEF != 0.0:
            current_nnd = []
            nearest_indices: list[int | None] = []
            for i, d in enumerate(self.drones):
                nearest_idx = None
                nnd = float("inf")
                for j, d2 in enumerate(self.drones):
                    if j == i:
                        continue
                    dist = math.hypot(d.position.x - d2.position.x, d.position.y - d2.position.y)
                    if dist < nnd:
                        nnd = dist
                        nearest_idx = j
                if nearest_idx is None:
                    nnd = 0.0
                current_nnd.append(nnd)
                nearest_indices.append(nearest_idx)

            for i in range(n):
                prev = self._prev_agent_nndists[i] if i < len(self._prev_agent_nndists) else current_nnd[i]
                delta = current_nnd[i] - prev  # positive = moved away
                shaped = max(-DISPERSION_SHAPING_CLIP, min(DISPERSION_SHAPING_CLIP, (delta) / WORLD_SCALE))
                rewards[i] += DISPERSION_REWARD_COEF * shaped

                separation = min(1.0, current_nnd[i] / max(1.0, INITIAL_SEPARATION_TARGET_DIST))
                rewards[i] += INITIAL_SEPARATION_REWARD_COEF * separation

                cell = self._grid_cell_for_pos(self.drones[i].position.x, self.drones[i].position.y)
                cell_count = sum(
                    1
                    for d in self.drones
                    if self._grid_cell_for_pos(d.position.x, d.position.y) == cell
                )
                rewards[i] += INITIAL_UNIQUE_GRID_REWARD / max(1, cell_count)
                is_outer_cell = (
                    cell[0] == 0
                    or cell[0] == GRID_ROWS - 1
                    or cell[1] == 0
                    or cell[1] == GRID_COLS - 1
                )
                if is_outer_cell:
                    rewards[i] += INITIAL_OUTER_GRID_REWARD
                elif cell == (GRID_ROWS // 2, GRID_COLS // 2) and self._known_prey_grid_cell is None:
                    rewards[i] += INITIAL_CENTER_GRID_PENALTY

                nearest_idx = nearest_indices[i]
                if nearest_idx is not None:
                    away = self.drones[i].position - self.drones[nearest_idx].position
                    if away.length_squared() > 1e-6 and self.drones[i].velocity.length_squared() > 1e-6:
                        away = away.normalize()
                        speed_along_away = self.drones[i].velocity.dot(away) / DRONE_SPEED
                        rewards[i] += AWAY_FROM_TEAMMATE_REWARD_COEF * max(0.0, speed_along_away)

            self._prev_agent_nndists = current_nnd

        # See-prey incentives: reward agents that have the prey in sensor range
        # and give a team bonus if a majority (configurable fraction) can see it.
        if self.prey is not None:
            px, py = self.prey.position.x, self.prey.position.y
            see_counts = 0
            for i, d in enumerate(self.drones):
                dist = math.hypot(d.position.x - px, d.position.y - py)
                if dist <= R_SENSE:
                    rewards[i] += SEE_PREY_REWARD
                    see_counts += 1
            if see_counts >= math.ceil(SEE_PREY_TEAM_FRAC * max(1, n)):
                for i in range(n):
                    rewards[i] += SEE_PREY_TEAM_BONUS

        # explicit flank/contributor reward: give an extra per-step reward to
        # agents that are contributors according to the capture geometry.
        if CONTRIBUTOR_BONUS_ENABLED:
            for idx in gap.contributor_indices:
                if 0 <= idx < n:
                    # keep the small contributor bonus already applied above,
                    # and add a dedicated flank reward to emphasize correct
                    # flanking positions during learning.
                    rewards[idx] += FLANK_REWARD

        # optional per-agent contributor bonus
        if CONTRIBUTOR_BONUS_ENABLED:
            for idx in gap.contributor_indices:
                if 0 <= idx < n:
                    rewards[idx] += CONTRIBUTOR_BONUS

        self._apply_grid_rewards(rewards, gap)

        return rewards

    def _apply_grid_rewards(self, rewards: dict[int, float], gap: GapResult) -> None:
        n = len(self.drones)
        if n == 0:
            return

        occupied = self._grid_cells_occupied_by_drones()
        new_cells = set(occupied) - self._visited_grid_cells
        for i, d in enumerate(self.drones):
            cell = self._grid_cell_for_pos(d.position.x, d.position.y)
            if cell in new_cells:
                rewards[i] += GRID_DISCOVERY_REWARD
        self._visited_grid_cells.update(occupied)

        coverage_frac = len(occupied) / max(1, GRID_ROWS * GRID_COLS)
        for i in range(n):
            rewards[i] += GRID_COVERAGE_REWARD * coverage_frac

        target_cell: tuple[int, int] | None = self._known_prey_grid_cell
        if self.prey is not None:
            prey_cell = self._grid_cell_for_pos(self.prey.position.x, self.prey.position.y)
            same_cell_agents = [
                i for i, d in enumerate(self.drones)
                if self._grid_cell_for_pos(d.position.x, d.position.y) == prey_cell
            ]
            if same_cell_agents:
                self._known_prey_grid_cell = prey_cell
                target_cell = prey_cell
                for i in same_cell_agents:
                    rewards[i] += PREY_GRID_DISCOVERY_REWARD

        if target_cell is not None and GRID_CONVERGE_REWARD_COEF != 0.0:
            target = self._grid_center(target_cell)
            current_dists = [d.position.distance_to(target) for d in self.drones]
            for i, dist in enumerate(current_dists):
                prev = (
                    self._prev_agent_grid_target_dists[i]
                    if i < len(self._prev_agent_grid_target_dists)
                    else dist
                )
                delta = prev - dist
                shaped = max(
                    -GRID_CONVERGE_SHAPING_CLIP,
                    min(GRID_CONVERGE_SHAPING_CLIP, delta / WORLD_SCALE),
                )
                rewards[i] += GRID_CONVERGE_REWARD_COEF * shaped
            self._prev_agent_grid_target_dists = current_dists
        elif len(self._grid_search_targets) == n:
            self._prev_agent_grid_target_dists = [
                d.position.distance_to(self._grid_search_targets[i])
                for i, d in enumerate(self.drones)
            ]

        self._apply_flank_diversity_rewards(rewards, gap)

    def _apply_flank_diversity_rewards(self, rewards: dict[int, float], gap: GapResult) -> None:
        if self.prey is None or len(gap.contributor_indices) < 2:
            return

        px, py = self.prey.position.x, self.prey.position.y
        angles: dict[int, float] = {}
        for idx in gap.contributor_indices:
            if 0 <= idx < len(self.drones):
                d = self.drones[idx]
                angles[idx] = math.atan2(d.position.y - py, d.position.x - px)
        if len(angles) < 2:
            return

        for idx, angle in angles.items():
            nearest_sep = min(
                abs((angle - other + math.pi) % (2 * math.pi) - math.pi)
                for other_idx, other in angles.items()
                if other_idx != idx
            )
            rewards[idx] += FLANK_DIVERSITY_REWARD * min(1.0, nearest_sep / (math.pi / 2))

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
        self._draw_grid(screen)
        for obs in self.obstacles:
            obs.draw(screen)
        for drone in self.drones:
            drone.draw(screen)
        if self.prey is not None:
            self.prey.draw(screen)

        # optional visualization: draw flank targets and lines from predators
        try:
            if self._last_flank_targets and len(self._last_flank_targets) == len(self.drones):
                for drone, tgt in zip(self.drones, self._last_flank_targets):
                    # line from predator to flank target
                    pygame.draw.line(screen, (200, 100, 100), (drone.position.x, drone.position.y), (tgt.x, tgt.y), 1)
                    # small circle at flank target
                    pygame.draw.circle(screen, (200, 100, 100), (int(tgt.x), int(tgt.y)), 4, 1)
        except Exception:
            # rendering should never crash the environment; swallow any errors
            pass

        try:
            if self._last_grid_targets and len(self._last_grid_targets) == len(self.drones):
                for drone, tgt in zip(self.drones, self._last_grid_targets):
                    pygame.draw.line(screen, (120, 160, 210), (drone.position.x, drone.position.y), (tgt.x, tgt.y), 1)
                    pygame.draw.circle(screen, (120, 160, 210), (int(tgt.x), int(tgt.y)), 3, 1)
        except Exception:
            pass

    def _draw_grid(self, screen: pygame.Surface) -> None:
        cell_w = self._width / GRID_COLS
        cell_h = self._height / GRID_ROWS
        color = (75, 80, 95)
        for c in range(1, GRID_COLS):
            x = round(c * cell_w)
            pygame.draw.line(screen, color, (x, 0), (x, self._height), 1)
        for r in range(1, GRID_ROWS):
            y = round(r * cell_h)
            pygame.draw.line(screen, color, (0, y), (self._width, y), 1)

        if self._known_prey_grid_cell is not None:
            row, col = self._known_prey_grid_cell
            rect = pygame.Rect(round(col * cell_w), round(row * cell_h), round(cell_w), round(cell_h))
            pygame.draw.rect(screen, (120, 120, 60), rect, 2)

    # ── accessors (used by wrapper / tests) ───────────────────────────────

    @property
    def num_agents(self) -> int:
        return len(self.drones)

    @property
    def obs_size(self) -> int:
        return OBS_SIZE
