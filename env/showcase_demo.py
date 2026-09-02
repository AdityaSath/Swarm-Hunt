"""Reliable scripted swarm-capture showcase.

This demo intentionally uses a deterministic formation controller rather than
an RL checkpoint. It exercises the real environment, movement, collision, and
capture rules while providing a dependable visual baseline.
"""

from __future__ import annotations

import argparse

import pygame

from swarm_env.capture import CaptureStatus, PreyTacticalState
from swarm_env.config import (
    ARENA_HEIGHT,
    ARENA_WIDTH,
    CAPTURE_HOLD_STEPS,
    COMBO_CAPTURE_NEED,
    DT,
    FPS,
    R_CAPTURE_RANGE,
)
from swarm_env.environment import Environment
from swarm_env.formation_controller import FormationController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scripted swarm capture showcase")
    parser.add_argument("--prey-speed-factor", type=float, default=0.5)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional first episode seed; later resets use seed+1, seed+2, ...",
    )
    parser.add_argument("--episodes", type=int, default=0,
                        help="Exit after N episodes (0 = run until closed)")
    parser.add_argument("--no-obstacles", action="store_true",
                        help="Use an open arena instead of the default obstacle layout")
    parser.add_argument("--show-targets", action="store_true",
                        help="Draw each drone's assigned formation target")
    return parser.parse_args()


def reset_showcase(
    env: Environment,
    controller: FormationController,
    seed: int | None,
    no_obstacles: bool,
) -> None:
    env.reset(seed=seed)
    if no_obstacles:
        env.obstacles.clear()
    controller.reset(env)


def main() -> None:
    args = parse_args()
    pygame.init()
    screen = pygame.display.set_mode((ARENA_WIDTH, ARENA_HEIGHT))
    pygame.display.set_caption("Swarm Hunt - Scripted Capture Showcase")
    clock = pygame.time.Clock()
    title_font = pygame.font.Font(None, 42)
    hud_font = pygame.font.Font(None, 25)

    env = Environment(dt=DT, prey_speed_factor=args.prey_speed_factor)
    controller = FormationController()
    reset_index = 0
    current_seed = args.seed

    def start_next_episode() -> None:
        nonlocal reset_index, current_seed, capture, tactical
        current_seed = args.seed + reset_index if args.seed is not None else None
        reset_index += 1
        reset_showcase(env, controller, current_seed, args.no_obstacles)
        capture = CaptureStatus(0, [], 0, 0)
        tactical = PreyTacticalState.FREE

    paused = False
    running = True
    episode = 0
    captures = 0
    reset_frames = 0
    capture = CaptureStatus(0, [], 0, 0)
    tactical = PreyTacticalState.FREE
    start_next_episode()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    start_next_episode()
                    reset_frames = 0

        if not paused:
            if reset_frames > 0:
                reset_frames -= 1
                if reset_frames == 0:
                    if args.episodes > 0 and episode >= args.episodes:
                        running = False
                    else:
                        start_next_episode()
            else:
                actions = controller.actions(env)
                _, _, terminations, truncations, infos = env.step(actions)
                capture = infos["capture"]
                tactical = infos["tactical_state"]
                if any(terminations.values()) or any(truncations.values()):
                    episode += 1
                    if any(terminations.values()):
                        captures += 1
                    reset_frames = int(1.5 * FPS)

        screen.fill((30, 30, 40))
        env.render(screen)

        if env.prey is not None:
            prey_center = (round(env.prey.position.x), round(env.prey.position.y))
            pygame.draw.circle(
                screen, (70, 135, 220), prey_center, round(R_CAPTURE_RANGE), 2
            )

        if args.show_targets:
            for agent_idx, target in controller.slot_targets(env).items():
                color = (110, 180, 230)
                pygame.draw.line(screen, color, env.drones[agent_idx].position, target, 1)
                pygame.draw.circle(screen, color, (round(target.x), round(target.y)), 5, 1)

        combined = capture.wall_count + capture.in_range_count
        hold_pct = min(100.0, 100.0 * capture.hold_counter / max(1, CAPTURE_HOLD_STEPS))

        title_color = {
            PreyTacticalState.FREE: (110, 220, 140),
            PreyTacticalState.THREATENED: (255, 215, 90),
            PreyTacticalState.CAPTURED: (255, 90, 90),
        }[tactical]
        title = title_font.render(
            "CAPTURED" if tactical == PreyTacticalState.CAPTURED else tactical.name,
            True,
            title_color,
        )
        screen.blit(title, (ARENA_WIDTH // 2 - title.get_width() // 2, 14))

        hud_lines = (
            "SCRIPTED FORMATION BASELINE (not a trained policy)",
            f"Capture: {capture.in_range_count} drones + {capture.wall_count} walls "
            f"= {combined}/{COMBO_CAPTURE_NEED}  |  hold {hold_pct:.0f}%",
            f"Captures: {captures}/{episode}  |  prey speed: "
            f"{args.prey_speed_factor:.2f}x  |  seed: "
            f"{current_seed if current_seed is not None else 'random'}  |  "
            "Space pause  R reset  Esc quit",
        )
        for row, line in enumerate(hud_lines):
            surface = hud_font.render(line, True, (225, 225, 230))
            screen.blit(surface, (12, ARENA_HEIGHT - 78 + row * 24))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
