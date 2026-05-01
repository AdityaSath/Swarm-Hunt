import os, sys, math
sys.path.insert(0, os.path.join(os.getcwd(), "env"))
from swarm_env.environment import Environment

def main():
    env = Environment()
    obs, info = env.reset()
    for t in range(400):
        obs, rewards, terms, truncs, info = env.step()
        g = info.get('gap')
        if g is not None and g.predator_contributors >= 3:
            print('t=', t, 'contributors=', g.predator_contributors, 'largest_gap_deg=', g.largest_gap*180/math.pi)
            px, py = env.prey.position.x, env.prey.position.y
            angles = [(i, round(math.degrees(math.atan2(d.position.y-py, d.position.x-px)),1)) for i,d in enumerate(env.drones)]
            print('pred angles (deg):', angles)
            dists = [(i, round(math.hypot(d.position.x-px, d.position.y-py),1)) for i,d in enumerate(env.drones)]
            print('pred dists:', dists)
            break
    print('done')

if __name__ == '__main__':
    main()
