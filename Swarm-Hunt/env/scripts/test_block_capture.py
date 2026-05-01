from swarm_env.environment import Environment
from swarm_env.config import R_CAP, PREY_RADIUS
import math


env = Environment()
# place prey at center
px = env._width / 2
py = env._height / 2
env.prey.position.x = px
env.prey.position.y = py
# place 4 predators evenly around at 0.5 * R_CAP
dist = 0.5 * R_CAP
n = len(env.drones)
for i in range(n):
    d = env.drones[i]
    ang = 2*math.pi*i/n
    d.position.x = px + dist*math.cos(ang)
    d.position.y = py + dist*math.sin(ang)
    d.velocity.x = 0
    d.velocity.y = 0

# step repeatedly to let FSM / capture logic run
captured = False
for t in range(40):
    obs, rew, terms, truncs, info = env.step(None)
    tactical = info.get('tactical_state')
    print(f"t={t+1} tactical={tactical}")
    if getattr(tactical, 'name', None) == 'CAPTURED':
        print('Captured at step', t+1)
        captured = True
        break

print('Result captured=', captured)
