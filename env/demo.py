"""
Pygame demo using a trained MATD3 checkpoint.

Usage:
    python demo.py                                          # most recent .pt
    python demo.py --checkpoint models/MATD3/some_file.pt
    python demo.py --prey-speed-factor 0.5                  # easier prey
    python demo.py --action-repeat 2                        # match training (default)
    python demo.py --prey-bounce-scale 0.5                   # override bounce speed scale
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pygame
import torch

from swarm_env.config import (
    ARENA_WIDTH,
    ARENA_HEIGHT,
    FPS,
    DT,
    DRONE_COUNT,
    PREY_SPEED,
    PREY_BOUNCE_SPEED_SCALE,
)
from swarm_env.environment import Environment
from swarm_env.capture import PreyTacticalState
from swarm_ml import AGENT_IDS, build_matd3, load_bc_actors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo trained MATD3 agents")
    policy = p.add_mutually_exclusive_group()
    policy.add_argument("--checkpoint", type=str, default=None,
                        help="MATD3 .pt (default: most recent in models/MATD3/)")
    policy.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Behavior-cloned actor from pretrain_bc.py")
    p.add_argument("--prey-speed-factor", type=float, default=1.0)
    p.add_argument("--action-repeat", type=int, default=2,
                   help="Physics steps per policy decision (match train.py)")
    p.add_argument("--episodes", type=int, default=0,
                   help="Auto-reset after N episodes (0 = infinite, manual R to reset)")
    p.add_argument(
        "--prey-bounce-scale",
        type=float,
        default=None,
        help="Bounce speed = PREY_SPEED * prey_speed_factor * this "
        f"(default: {PREY_BOUNCE_SPEED_SCALE} from config). "
        "Try 0.4–0.8 if motion looks frozen.",
    )
    return p.parse_args()


def build_agent(checkpoint: str, device: torch.device):
    agent = build_matd3(device=device)
    agent.load_checkpoint(checkpoint)
    # Eval mode: training=True makes get_action() add exploration noise (TD3/MATD3).
    agent.set_training_mode(False)
    return agent


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = args.checkpoint
    if checkpoint is None and args.bc_checkpoint is None:
        pts = sorted(glob.glob("./models/MATD3/*.pt"), key=os.path.getmtime)
        if not pts:
            print("No checkpoint found in models/MATD3/. Train first.")
            return
        checkpoint = pts[-1]

    if args.bc_checkpoint:
        agent = build_matd3(device=device)
        load_bc_actors(agent, args.bc_checkpoint)
        loaded_path = args.bc_checkpoint
        policy_label = "behavior-cloned actor"
    else:
        assert checkpoint is not None
        agent = build_agent(checkpoint, device)
        loaded_path = checkpoint
        policy_label = "MATD3 checkpoint"
    agent.set_training_mode(False)
    print(f"Loaded {policy_label}: {loaded_path}  |  Device: {device}")

    _bs = args.prey_bounce_scale if args.prey_bounce_scale is not None else PREY_BOUNCE_SPEED_SCALE
    v = PREY_SPEED * args.prey_speed_factor * _bs
    print(
        f"Prey: wall-bouncing ball  |  speed ~ {v:.1f} px/s  (scale {_bs})"
    )

    pygame.init()
    screen = pygame.display.set_mode((ARENA_WIDTH, ARENA_HEIGHT))
    pygame.display.set_caption("Pursuit V1 - Trained Agent Demo")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    env = Environment(
        dt=DT,
        prey_speed_factor=args.prey_speed_factor,
        prey_bounce_speed_scale=args.prey_bounce_scale,
    )
    agent_ids = AGENT_IDS
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
            PreyTacticalState.CAPTURED: (255, 80, 80),
        }
        state_text = small_font.render(
            f"Tactical: {tactical.name}  |  Step: {env._step_count}  |  "
            f"Prey: {args.prey_speed_factor:.1f}x  |  "
            f"action_repeat: {args.action_repeat}",
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
