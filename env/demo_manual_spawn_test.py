"""
Test harness: prey at arena center (scripted policy off — only drones move it),
three static predators beside it, one predator controlled with WASD.

Run from repo root / env folder:
    python demo_manual_spawn_test.py

Uses the same capture rules as ``Environment`` (combo: walls in blue circle +
drones in ``R_CAPTURE_RANGE`` ≥ ``COMBO_CAPTURE_NEED``). Only extra behavior
here is **pushable prey** via ``ManualPushDemoEnv`` (overlap resolution).

Blue ring = R_CAPTURE_RANGE (1.2× legacy R_CAP).
"""

from __future__ import annotations

import math

import pygame

from swarm_env.capture import EpisodeState, PreyTacticalState
from swarm_env.config import (
    ARENA_WIDTH,
    ARENA_HEIGHT,
    FPS,
    DT,
    PREY_RADIUS,
    DRONE_RADIUS,
    DRONE_SPEED,
    PREY_SPEED,
    R_CAPTURE_RANGE,
    CAPTURE_HOLD_STEPS,
    COMBO_CAPTURE_NEED,
)
from swarm_env.environment import Environment

N_DRONES = 4
STATIC_COUNT = 3
PLAYER_IDX = 3


class ManualPushDemoEnv(Environment):
    """
    Same rules as ``Environment``, plus after each physics substep resolves
    predator–prey overlap by pushing **only the prey** out along the separation
    normal (slide velocity + damping).
    """

    def _physics_step(self) -> None:
        super()._physics_step()
        if self.prey is None:
            return
        pr = self.prey
        dt = self.dt
        for _ in range(16):
            any_hit = False
            for d in self.drones:
                dx = pr.position.x - d.position.x
                dy = pr.position.y - d.position.y
                dist = math.hypot(dx, dy)
                rsum = pr.radius + d.radius
                if dist >= rsum:
                    continue
                if dist < 1e-6:
                    dx, dy = 1.0, 0.0
                    dist = 1.0
                nx, ny = dx / dist, dy / dist
                overlap = rsum - dist
                pr.position.x += nx * overlap
                pr.position.y += ny * overlap
                sp = min(PREY_SPEED, (overlap / max(dt, 1e-9)) * 0.4)
                pr.velocity.x = nx * sp
                pr.velocity.y = ny * sp
                any_hit = True
            if not any_hit:
                break

        clamped = self.arena.clamp(pr.position, pr.radius)
        if clamped.x != pr.position.x or clamped.y != pr.position.y:
            pr.velocity = pygame.math.Vector2(0.0, 0.0)
        pr.position = clamped
        pr.velocity *= 0.96


def _separation_dist() -> float:
    """Center distance so prey and predator disks barely clear (no overlap)."""
    return float(PREY_RADIUS + DRONE_RADIUS + 3.0)


def apply_test_layout(env: Environment) -> None:
    """No obstacles; prey centered; three bots on a ring; player offset."""
    env.obstacles.clear()

    cx = env._width / 2.0
    cy = env._height / 2.0
    r_off = _separation_dist()

    if env.prey is not None:
        env.prey.position.update(cx, cy)
        env.prey.velocity.update(0.0, 0.0)
        env.prey.decide = lambda *a, **k: None  # noqa: ARG005

    for i in range(STATIC_COUNT):
        ang = i * (2.0 * math.pi / 3.0) - math.pi / 2.0
        env.drones[i].position.update(
            cx + math.cos(ang) * r_off,
            cy + math.sin(ang) * r_off,
        )
        env.drones[i].velocity.update(0.0, 0.0)

    p_ang = math.pi / 4.0
    p_dist = r_off * 1.85
    env.drones[PLAYER_IDX].position.update(
        cx + math.cos(p_ang) * p_dist,
        cy + math.sin(p_ang) * p_dist,
    )
    env.drones[PLAYER_IDX].velocity.update(0.0, 0.0)

    env._fsm.reset()
    env._episode_state = EpisodeState.IN_PURSUIT
    env._step_count = 0
    env._prev_predator_distances = env._pred_prey_distances()
    env._prev_tactical = PreyTacticalState.FREE
    env._obs_collisions = [False] * len(env.drones)
    env._pred_collisions = [False] * len(env.drones)


def _wasd_velocity(keys) -> tuple[float, float]:
    vx = 0.0
    vy = 0.0
    if keys[pygame.K_w]:
        vy -= 1.0
    if keys[pygame.K_s]:
        vy += 1.0
    if keys[pygame.K_a]:
        vx -= 1.0
    if keys[pygame.K_d]:
        vx += 1.0
    if vx == 0.0 and vy == 0.0:
        return 0.0, 0.0
    length = math.hypot(vx, vy)
    return DRONE_SPEED * vx / length, DRONE_SPEED * vy / length


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((ARENA_WIDTH, ARENA_HEIGHT))
    pygame.display.set_caption("Manual spawn test — WASD = yellow drone, R = reset layout")
    clock = pygame.time.Clock()
    hud_small = pygame.font.Font(None, 22)
    banner_font = pygame.font.Font(None, 96)
    sub_font = pygame.font.Font(None, 34)

    env = ManualPushDemoEnv(dt=DT, drone_count=N_DRONES, prey_speed_factor=0.0)
    env.reset()
    apply_test_layout(env)

    static_color = (110, 110, 130)
    player_color = (240, 210, 80)

    state_colors = {
        PreyTacticalState.FREE: ((90, 200, 120), (25, 55, 30)),
        PreyTacticalState.THREATENED: ((255, 210, 90), (60, 45, 15)),
        PreyTacticalState.CAPTURED: ((255, 60, 60), (80, 15, 15)),
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    apply_test_layout(env)

        keys = pygame.key.get_pressed()
        pvx, pvy = _wasd_velocity(keys)
        actions = {
            0: (0.0, 0.0),
            1: (0.0, 0.0),
            2: (0.0, 0.0),
            PLAYER_IDX: (pvx, pvy),
        }
        _, _, _, _, infos = env.step(actions)
        cap = infos["capture"]
        tact: PreyTacticalState = infos["tactical_state"]
        ep = infos["episode_state"]

        screen.fill((30, 30, 40))
        env.render(screen)
        if env.prey is not None:
            px = int(env.prey.position.x)
            py = int(env.prey.position.y)
            pygame.draw.circle(screen, (80, 120, 180), (px, py), int(R_CAPTURE_RANGE), 2)

        for i in range(STATIC_COUNT):
            env.drones[i].draw(screen, static_color)
        env.drones[PLAYER_IDX].draw(screen, player_color)

        fg, bg = state_colors.get(tact, ((200, 200, 200), (40, 40, 50)))
        label = tact.name
        if tact == PreyTacticalState.CAPTURED or ep == EpisodeState.CAPTURED:
            overlay = pygame.Surface((ARENA_WIDTH, ARENA_HEIGHT), pygame.SRCALPHA)
            overlay.fill((120, 20, 20, 110))
            screen.blit(overlay, (0, 0))

        title = banner_font.render(label, True, fg)
        tw = title.get_width()
        pad = 16
        bar = pygame.Surface((tw + pad * 2, title.get_height() + pad))
        bar.set_alpha(230)
        bar.fill(bg)
        bx = ARENA_WIDTH // 2 - (tw + pad * 2) // 2
        by = 28
        screen.blit(bar, (bx, by - pad // 2))
        screen.blit(title, (ARENA_WIDTH // 2 - tw // 2, by))

        hold_pct = 100.0 * cap.hold_counter / max(1, CAPTURE_HOLD_STEPS)
        wcount = cap.wall_count
        combined = wcount + cap.in_range_count
        timer_ok = combined >= COMBO_CAPTURE_NEED
        sub_lines = [
            f"Combo: {wcount} walls + {cap.in_range_count} drones = {combined} "
            f"(need >= {COMBO_CAPTURE_NEED} for timer)  {'ON' if timer_ok else 'off'}",
            f"Hold {cap.hold_counter} / {CAPTURE_HOLD_STEPS} ({hold_pct:.0f}%)  |  "
            f"episode: {ep.name}  |  step {env._step_count}",
        ]
        for i, line in enumerate(sub_lines):
            s = sub_font.render(line, True, (235, 235, 235))
            screen.blit(s, (ARENA_WIDTH // 2 - s.get_width() // 2, by + 100 + i * 38))

        if tact == PreyTacticalState.CAPTURED:
            boom = banner_font.render("CAPTURED", True, (255, 255, 100))
            screen.blit(
                boom,
                (ARENA_WIDTH // 2 - boom.get_width() // 2, ARENA_HEIGHT // 2 - 60),
            )

        help_lines = [
            "WASD: yellow drone  |  push prey with drones  |  R: reset  |  Esc: quit",
            f"Same capture as training: walls+drones in blue circle (r={R_CAPTURE_RANGE:.0f}) "
            f"≥ {COMBO_CAPTURE_NEED}; hold ~2 s.",
        ]
        for row, line in enumerate(help_lines):
            surf = hud_small.render(line, True, (180, 180, 190))
            screen.blit(surf, (12, ARENA_HEIGHT - 52 + row * 22))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
