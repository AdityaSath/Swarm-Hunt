"""Visual runner for scripted or SB3 policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import PPO

from locked_policy import LockedPolicy
from swarm_env.config import ARENA_HEIGHT, ARENA_WIDTH, FPS
from swarm_env.environment import Environment


def _flatten_obs(obs: dict[int, np.ndarray], n_agents: int) -> np.ndarray:
    return np.concatenate([obs[i].astype(np.float32) for i in range(n_agents)], axis=0)


def _vector_to_actions(action: np.ndarray, n_agents: int) -> dict[int, tuple[float, float]]:
    return {
        i: (float(action[2 * i]), float(action[2 * i + 1]))
        for i in range(n_agents)
    }


def _load_policy(args: argparse.Namespace):
    if args.manifest:
        return LockedPolicy(args.manifest), "manifest"
    return PPO.load(args.model), "sb3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_warm_v1.zip", help="Path to an SB3 PPO .zip")
    parser.add_argument("--manifest", help="Path to a locked policy manifest")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--hybrid-actions", action="store_true", help="Interpret model output as high-level hybrid intent")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((ARENA_WIDTH, ARENA_HEIGHT))
    pygame.display.set_caption("V1 Pursuit Environment - Policy Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)

    policy, policy_type = _load_policy(args)
    env = Environment()
    obs, _ = env.reset()
    paused = False
    episode_over = False
    episode_msg = ""

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
                    episode_msg = ""

        if not paused and not episode_over:
            if policy_type == "manifest":
                if policy.type == "scripted":
                    action = policy.predict(None, repo_env=env, deterministic=args.deterministic)
                else:
                    action = policy.predict(
                        _flatten_obs(obs, env.num_agents),
                        repo_env=env,
                        deterministic=args.deterministic,
                    )
            else:
                action, _ = policy.predict(
                    _flatten_obs(obs, env.num_agents),
                    deterministic=args.deterministic,
                )

            if args.hybrid_actions:
                env_actions = env.hybrid_actions(np.asarray(action))
            else:
                env_actions = _vector_to_actions(np.asarray(action), env.num_agents)
            obs, _, terms, truncs, _ = env.step(env_actions)
            if any(terms.values()):
                episode_over = True
                episode_msg = "CAPTURED - press R to restart"
            elif any(truncs.values()):
                episode_over = True
                episode_msg = "TIMEOUT - press R to restart"

        screen.fill((30, 30, 40))
        env.render(screen)

        label = "RL policy"
        if policy_type == "manifest":
            label = f"Locked {policy.type}: {Path(args.manifest).stem}"
        elif args.model:
            label = f"SB3: {Path(args.model).name}"
        text = font.render(label, True, (210, 210, 210))
        screen.blit(text, (12, 10))

        if paused:
            text = font.render("PAUSED", True, (200, 200, 200))
            screen.blit(text, (ARENA_WIDTH // 2 - 50, 10))

        if episode_over:
            text = font.render(episode_msg, True, (255, 220, 100))
            screen.blit(text, (ARENA_WIDTH // 2 - text.get_width() // 2, 44))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
