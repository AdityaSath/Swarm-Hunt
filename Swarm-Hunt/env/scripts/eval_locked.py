"""Evaluate a locked policy manifest over multiple episodes.

Usage:
  PYTHONPATH="$(pwd)/env" .venv/bin/python env/scripts/eval_locked.py --manifest env/models/scripted_v1.json --episodes 10
"""
import argparse
import time
import numpy as np

from locked_policy import LockedPolicy, get_manifest_path
from swarm_env.environment import Environment


def run_eval(manifest_path: str, episodes: int = 10, max_steps: int = 2000):
    lp = LockedPolicy(manifest_path)
    env = Environment()
    captures = 0
    truncs = 0
    lengths = []
    for ep in range(episodes):
        obs, info = env.reset()
        for t in range(max_steps):
            # construct centralized obs for SB3 if needed (flatten)
            # LockedPolicy will call repo_env.scripted_actions() when scripted; for SB3 we supply flattened obs
            if lp.type == 'scripted':
                action = lp.predict(None, repo_env=env)
            else:
                # flatten obs in same ordering as CentralizedSwarmGym
                obs_vec = np.concatenate([obs[i].astype(np.float32) for i in range(env.num_agents)], axis=0)
                action = lp.predict(obs_vec, repo_env=env)

            # convert action vector back to per-agent dict
            actions = {i: (float(action[2*i]), float(action[2*i+1])) for i in range(env.num_agents)}
            obs, rewards, terms, truncs_dict, info = env.step(actions)
            if any(terms.values()):
                captures += 1
                lengths.append(t+1)
                break
            if any(truncs_dict.values()):
                truncs += 1
                lengths.append(t+1)
                break
        else:
            lengths.append(max_steps)
    print(f'episodes={episodes}, captures={captures}, truncations={truncs}, mean_len={np.mean(lengths):.1f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=2000)
    args = parser.parse_args()
    run_eval(args.manifest, args.episodes, args.max_steps)


if __name__ == '__main__':
    main()
