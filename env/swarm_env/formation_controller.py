"""Deterministic formation controller used by the scripted showcase demo.

This is a baseline controller, not a learned policy.  It gives each predator a
stable slot around a short-horizon prediction of the prey, which makes the
capture task easy to demonstrate and provides a useful target for RL training.
"""

from __future__ import annotations

import math

import pygame

from swarm_env.config import (
    DRONE_RADIUS,
    DRONE_SPEED,
    R_CAPTURE_RANGE,
)
from swarm_env.environment import Environment


class FormationController:
    """Drive predators into evenly spaced slots around the moving prey."""

    def __init__(
        self,
        ring_radius: float = R_CAPTURE_RANGE * 0.72,
        lookahead_seconds: float = 0.45,
        position_gain: float = 1.8,
        separation_distance: float = DRONE_RADIUS * 3.2,
    ) -> None:
        self.ring_radius = min(float(ring_radius), R_CAPTURE_RANGE * 0.9)
        self.lookahead_seconds = max(0.0, float(lookahead_seconds))
        self.position_gain = max(0.0, float(position_gain))
        self.separation_distance = max(DRONE_RADIUS * 2.0, float(separation_distance))
        self._slot_for_agent: dict[int, int] = {}

    def reset(self, env: Environment) -> None:
        """Use the same stable role assignment exposed in observations."""
        self._slot_for_agent = {i: i for i in range(len(env.drones))}

    def actions(self, env: Environment) -> dict[int, tuple[float, float]]:
        """Return desired world-frame velocities for every predator."""
        if env.prey is None:
            return {i: (0.0, 0.0) for i in range(len(env.drones))}
        if len(self._slot_for_agent) != len(env.drones):
            self.reset(env)

        center, predicted_prey_velocity = self._predict_prey(env)
        slots = self._slot_positions(env, center)
        actions: dict[int, tuple[float, float]] = {}

        for agent_idx, drone in enumerate(env.drones):
            slot_idx = self._slot_for_agent[agent_idx]
            target = slots[slot_idx]
            error = target - drone.position

            # Prey velocity is feed-forward; proportional position feedback
            # supplies the closing/interception component.
            desired = predicted_prey_velocity + error * self.position_gain
            desired += self._separation_velocity(env, agent_idx)
            desired += self._obstacle_avoidance_velocity(env, agent_idx, desired)

            if desired.length_squared() > DRONE_SPEED * DRONE_SPEED:
                desired.scale_to_length(DRONE_SPEED)
            actions[agent_idx] = (float(desired.x), float(desired.y))

        return actions

    def slot_targets(self, env: Environment) -> dict[int, pygame.math.Vector2]:
        """Expose current targets for optional debug rendering."""
        if env.prey is None:
            return {}
        if len(self._slot_for_agent) != len(env.drones):
            self.reset(env)
        center, _ = self._predict_prey(env)
        slots = self._slot_positions(env, center)
        return {
            agent_idx: slots[slot_idx]
            for agent_idx, slot_idx in self._slot_for_agent.items()
        }

    def _predict_prey(
        self, env: Environment
    ) -> tuple[pygame.math.Vector2, pygame.math.Vector2]:
        assert env.prey is not None
        position = env.prey.position.copy()
        velocity = env.prey.velocity.copy()
        remaining = self.lookahead_seconds

        # Small simulation steps correctly predict one or more wall bounces.
        while remaining > 1e-9:
            dt = min(0.05, remaining)
            position += velocity * dt
            position, velocity = env.arena.clamp_and_bounce(
                position, velocity, env.prey.radius
            )
            remaining -= dt
        return position, velocity

    def _slot_positions(
        self, env: Environment, center: pygame.math.Vector2
    ) -> list[pygame.math.Vector2]:
        n = len(env.drones)
        x_min, y_min, x_max, y_max = env.arena.get_bounds()
        margin = DRONE_RADIUS + 1.0
        positions: list[pygame.math.Vector2] = []

        for slot_idx in range(n):
            angle = -math.pi / 2.0 + 2.0 * math.pi * slot_idx / max(1, n)
            target = center + pygame.math.Vector2(
                math.cos(angle) * self.ring_radius,
                math.sin(angle) * self.ring_radius,
            )
            target.x = max(x_min + margin, min(x_max - margin, target.x))
            target.y = max(y_min + margin, min(y_max - margin, target.y))
            positions.append(target)
        return positions

    def _separation_velocity(
        self, env: Environment, agent_idx: int
    ) -> pygame.math.Vector2:
        drone = env.drones[agent_idx]
        correction = pygame.math.Vector2()
        for other_idx, other in enumerate(env.drones):
            if other_idx == agent_idx:
                continue
            delta = drone.position - other.position
            distance = delta.length()
            if 1e-6 < distance < self.separation_distance:
                strength = 0.65 * DRONE_SPEED * (
                    1.0 - distance / self.separation_distance
                )
                correction += delta * (strength / distance)
        return correction

    def _obstacle_avoidance_velocity(
        self,
        env: Environment,
        agent_idx: int,
        desired: pygame.math.Vector2,
    ) -> pygame.math.Vector2:
        """Repel from nearby obstacles and imminent straight-line impacts."""
        if not env.obstacles:
            return pygame.math.Vector2()

        drone = env.drones[agent_idx]
        correction = pygame.math.Vector2()
        influence = 75.0
        lookahead = drone.position + desired.normalize() * 55.0 if desired else drone.position

        for obstacle in env.obstacles:
            rect = obstacle.get_collision_rect().inflate(
                int(2 * (DRONE_RADIUS + 5.0)), int(2 * (DRONE_RADIUS + 5.0))
            )
            closest = pygame.math.Vector2(
                max(rect.left, min(rect.right, drone.position.x)),
                max(rect.top, min(rect.bottom, drone.position.y)),
            )
            away = drone.position - closest
            distance = away.length()

            if distance < 1e-6 and rect.collidepoint(drone.position):
                distances = (
                    (abs(drone.position.x - rect.left), pygame.math.Vector2(-1.0, 0.0)),
                    (abs(rect.right - drone.position.x), pygame.math.Vector2(1.0, 0.0)),
                    (abs(drone.position.y - rect.top), pygame.math.Vector2(0.0, -1.0)),
                    (abs(rect.bottom - drone.position.y), pygame.math.Vector2(0.0, 1.0)),
                )
                away = min(distances, key=lambda item: item[0])[1]
                distance = 0.0

            imminent = rect.collidepoint(lookahead)
            if distance < influence or imminent:
                if away.length_squared() > 0.0:
                    away = away.normalize()
                strength = DRONE_SPEED * (
                    1.2 if imminent else 0.8 * (1.0 - distance / influence)
                )
                correction += away * strength

        return correction
