"""Entry point for the 2D swarm environment."""

import pygame

from swarm_env.environment import Environment
from swarm_env.config import ARENA_WIDTH, ARENA_HEIGHT, FPS


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((ARENA_WIDTH, ARENA_HEIGHT))
    pygame.display.set_caption("2D Swarm Environment")
    clock = pygame.time.Clock()

    env = Environment()
    paused = False

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

        if not paused:
            env.step()  # actions=None: drones keep current velocity (demo mode)

        screen.fill((30, 30, 40))
        env.render(screen)
        if paused:
            font = pygame.font.Font(None, 36)
            text = font.render("PAUSED", True, (200, 200, 200))
            screen.blit(text, (ARENA_WIDTH // 2 - 50, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
