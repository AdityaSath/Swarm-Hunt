"""Entry point for the V1 pursuit environment (Pygame demo)."""

import pygame

from swarm_env.environment import Environment
from swarm_env.config import ARENA_WIDTH, ARENA_HEIGHT, FPS, DT


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((ARENA_WIDTH, ARENA_HEIGHT))
    pygame.display.set_caption("V1 Pursuit Environment")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    env = Environment(dt=DT)
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
                    env.reset()
                    episode_over = False

        if not paused and not episode_over:
            _, _, terms, truncs, _ = env.step()
            if any(terms.values()):
                episode_over = True
                episode_msg = "CAPTURED - press R to restart"
            elif any(truncs.values()):
                episode_over = True
                episode_msg = "TIMEOUT - press R to restart"

        screen.fill((30, 30, 40))
        env.render(screen)

        if paused:
            text = font.render("PAUSED", True, (200, 200, 200))
            screen.blit(text, (ARENA_WIDTH // 2 - 50, 10))

        if episode_over:
            text = font.render(episode_msg, True, (255, 220, 100))
            screen.blit(text, (ARENA_WIDTH // 2 - text.get_width() // 2, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
