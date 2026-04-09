"""
Pygame demo using a trained MATD3 checkpoint.

Usage:
    python demo.py                                          # most recent .pt
    python demo.py --checkpoint models/MATD3/some_file.pt
    python demo.py --prey-speed-factor 0.5                  # easier prey
    python demo.py --action-repeat 4                        # match training (default)
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pygame
import torch
from agilerl.algorithms.matd3 import MATD3

from swarm_env.config import ARENA_WIDTH, ARENA_HEIGHT, FPS, DT, DRONE_COUNT, DRONE_SPEED
from swarm_env.environment import Environment, OBS_SIZE
from swarm_env.capture import PreyTacticalState
import gymnasium


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo trained MATD3 agents")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .pt file (default: most recent in models/MATD3/)")
    p.add_argument("--prey-speed-factor", type=float, default=1.0)
    p.add_argument("--action-repeat", type=int, default=4,
                   help="Physics steps per policy decision (match train.py)")
    p.add_argument("--episodes", type=int, default=0,
                   help="Auto-reset after N episodes (0 = infinite, manual R to reset)")
    return p.parse_args()


def build_agent(checkpoint: str, device: torch.device) -> MATD3:
    agent_ids = [f"predator_{i}" for i in range(DRONE_COUNT)]
    observation_spaces = [
        gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        for _ in agent_ids
    ]
    action_spaces = [
        gymnasium.spaces.Box(low=-DRONE_SPEED, high=DRONE_SPEED, shape=(2,), dtype=np.float32)
        for _ in agent_ids
    ]

    agent = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        device=device,
    )
    agent.load_checkpoint(checkpoint)
    return agent


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = args.checkpoint
    if checkpoint is None:
        pts = sorted(glob.glob("./models/MATD3/*.pt"), key=os.path.getmtime)
        if not pts:
            print("No checkpoint found in models/MATD3/. Train first.")
            return
        checkpoint = pts[-1]

    agent = build_agent(checkpoint, device)
    print(f"Loaded checkpoint: {checkpoint}  |  Device: {device}")

    pygame.init()
    screen = pygame.display.set_mode((ARENA_WIDTH, ARENA_HEIGHT))
    pygame.display.set_caption("Pursuit V1 — Trained Agent Demo")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    env = Environment(dt=DT, prey_speed_factor=args.prey_speed_factor)
    agent_ids = [f"predator_{i}" for i in range(DRONE_COUNT)]
    idx_to_agent = {i: a for i, a in enumerate(agent_ids)}

    episode = 0
    captures = 0
    episode_over = False
    episode_msg = ""
    paused = False
    repeat_left = 0
    cached_actions: dict[int, tuple[float, float]] | None = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    episode_over = False
                    repeat_left = 0
                    cached_actions = None

        if not paused and not episode_over:
            if repeat_left <= 0:
                obs_int = env._compute_observations()
                obs = {idx_to_agent[i]: v for i, v in obs_int.items()}
                cont_actions, _ = agent.get_action(obs)
                cached_actions = {}
                for i in range(DRONE_COUNT):
                    a = cont_actions[idx_to_agent[i]].reshape(-1)
                    cached_actions[i] = (float(a[0]), float(a[1]))
                repeat_left = max(1, args.action_repeat)

            assert cached_actions is not None
            _, _, terms, truncs, infos = env.step(cached_actions)
            repeat_left -= 1

            if any(terms.values()):
                episode_over = True
                episode += 1
                captures += 1
                episode_msg = f"CAPTURED!  ({captures}/{episode})  -  R to restart"
            elif any(truncs.values()):
                episode_over = True
                episode += 1
                episode_msg = f"TIMEOUT  ({captures}/{episode})  -  R to restart"

            if episode_over and args.episodes > 0 and episode >= args.episodes:
                running = False

        # ── render ────────────────────────────────────────────────────────
        screen.fill((30, 30, 40))
        env.render(screen)

        # HUD
        tactical = env._fsm.state
        state_colors = {
            PreyTacticalState.FREE: (100, 200, 100),
            PreyTacticalState.THREATENED: (220, 200, 80),
            PreyTacticalState.CONTAINED: (220, 140, 60),
            PreyTacticalState.CAPTURED: (255, 80, 80),
        }
        state_text = small_font.render(
            f"Tactical: {tactical.name}  |  Step: {env._step_count}  |  "
            f"Prey: {args.prey_speed_factor:.1f}x  |  action_repeat: {args.action_repeat}",
            True, state_colors.get(tactical, (200, 200, 200)),
        )
        screen.blit(state_text, (10, ARENA_HEIGHT - 30))

        if paused:
            text = font.render("PAUSED", True, (200, 200, 200))
            screen.blit(text, (ARENA_WIDTH // 2 - text.get_width() // 2, 10))

        if episode_over:
            text = font.render(episode_msg, True, (255, 220, 100))
            screen.blit(text, (ARENA_WIDTH // 2 - text.get_width() // 2, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    print(f"\nResults: {captures} captures / {episode} episodes")


if __name__ == "__main__":
    main()
