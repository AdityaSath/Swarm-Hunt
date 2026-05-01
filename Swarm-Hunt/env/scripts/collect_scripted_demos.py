"""Collect demonstrations using the scripted pure-pursuit baseline.

Saves actions and episode lengths to a npz file for later use (BC or analysis).

Run from repo root with PYTHONPATH pointing to `env`:
  PYTHONPATH="$(pwd)/env" .venv/bin/python env/scripts/collect_scripted_demos.py
"""

import argparse
import numpy as np
from swarm_env.environment import Environment


def collect(episodes: int = 50, max_steps: int = 2000, out: str = "scripted_demos.npz"):
    env = Environment()
    all_actions = []
    lengths = []
    for ep in range(episodes):
        obs, info = env.reset()
        ep_actions = []
        for t in range(max_steps):
            actions = env.scripted_actions()
            ep_actions.append([actions.get(i, (0.0, 0.0)) for i in range(env.num_agents)])
            obs, rewards, terms, truncs, info = env.step(actions)
            if any(terms.values()) or any(truncs.values()):
                lengths.append(t + 1)
                break
        else:
            lengths.append(max_steps)
        all_actions.append(np.array(ep_actions, dtype=np.float32))
        print(f"Collected episode {ep+1}/{episodes}, len={lengths[-1]}")

    # Save variable-length episodes as object array
    np.savez_compressed(out, actions=np.array(all_actions, dtype=object), lengths=np.array(lengths))
    print(f"Saved {episodes} episodes to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--out', type=str, default='scripted_demos.npz')
    args = parser.parse_args()
    collect(args.episodes, args.max_steps, args.out)
