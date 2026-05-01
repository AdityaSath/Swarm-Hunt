import sys, os, math
sys.path.insert(0, os.path.join(os.getcwd(), "env"))
from swarm_env.environment import Environment
from swarm_env.capture import PHI_ESCAPE_MAX

def main():
    env = Environment()
    obs, info = env.reset()
    print("PHI_ESCAPE_MAX (deg):", PHI_ESCAPE_MAX * 180.0 / math.pi)
    for t in range(400):
        obs, rewards, terms, truncs, info = env.step()
        if 'gap' in info and t % 20 == 0:
            gap = info['gap']
            lg = gap.largest_gap * 180.0 / math.pi
            contrib = gap.predator_contributors
            md = env._mean_pred_prey_dist()
            print("t=", t, "largest_gap_deg=", round(lg,1), "contributors=", contrib, "mean_dist=", round(md,1))
        if any(terms.values()):
            print("Captured at t=", t)
            break
    else:
        print("No capture within 400 steps")

if __name__ == '__main__':
    main()
