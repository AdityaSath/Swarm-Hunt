"""Visual evaluation runner for the centralized PPO policy.

This script loads a saved centralized PPO model (a single policy that outputs
concatenated desired velocities for all predators) and runs it inside the
existing Pygame demo loop so you can visually inspect behavior.

It is import-safe (no side-effects on import). Run with:

  .venv/bin/python env/evaluate_policy.py --model ./sb3_logs_smoke/ppo_central.zip

"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import pygame
import numpy as np

from gym_centralized import CentralizedSwarmGym
from swarm_env.config import ARENA_WIDTH, ARENA_HEIGHT, FPS


def run_visual_eval(model_path: str, deterministic: bool = True, max_steps: int | None = None) -> None:
    if not os.environ.get("MPLCONFIGDIR"):
        repo_root = Path(__file__).resolve().parents[1]
        mpl_dir = repo_root / ".mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
        os.environ.setdefault("MPLBACKEND", "Agg")

    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    env = CentralizedSwarmGym()

    # Start pygame window for rendering
    pygame.init()
    screen = pygame.display.set_mode((ARENA_WIDTH, ARENA_HEIGHT))
    pygame.display.set_caption("Evaluate Trained Centralized Policy")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    obs, _ = env.reset()
    paused = False
    episode_over = False
    episode_msg = ""
    steps = 0

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
                    obs, _ = env.reset()
                    episode_over = False

        if not paused and not episode_over:
            # Model expects the centralized observation vector
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                episode_over = True
                episode_msg = "TERMINATED - press R to restart"
            if max_steps is not None and steps >= max_steps:
                episode_over = True
                episode_msg = f"MAX STEPS ({max_steps}) - press R to restart"

        screen.fill((30, 30, 40))
        # Use underlying env rendering (CentralizedSwarmGym.env)
        env.env.render(screen)

        if paused:
            text = font.render("PAUSED", True, (200, 200, 200))
            screen.blit(text, (ARENA_WIDTH // 2 - 50, 10))

        if episode_over:
            text = font.render(episode_msg, True, (255, 220, 100))
            screen.blit(text, (ARENA_WIDTH // 2 - text.get_width() // 2, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved PPO model (.zip)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps per session")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_visual_eval(args.model, deterministic=args.deterministic, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
