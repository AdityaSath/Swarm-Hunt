import pygame
from swarm_env.environment import Environment

pygame.init()
# create an off-screen surface
surf = pygame.Surface((800, 600))
env = Environment(width=800, height=600)
obs, info = env.reset()
# run a few steps and render
for i in range(5):
    env.step()
    env.render(surf)
print('render ok')
pygame.quit()
