"""Run the scripted pure-pursuit baseline for multiple episodes and report capture counts.

Usage: run from repo root with PYTHONPATH pointing to the `env` package, e.g.
  PYTHONPATH="$(pwd)/env" .venv/bin/python env/scripts/eval_scripted.py
"""

from swarm_env.environment import Environment
from swarm_env.config import MAX_STEPS
import time


def run(episodes: int = 50, max_steps: int = MAX_STEPS):
    env = Environment()
    captures = 0
    truncs = 0
    times = []
    for ep in range(episodes):
        obs, info = env.reset()
        start = time.time()
        for t in range(max_steps):
            actions = env.scripted_actions()
            obs, rewards, terms, truncs_map, info = env.step(actions)
            if any(terms.values()):
                captures += 1
                times.append(t + 1)
                break
            if any(truncs_map.values()):
                truncs += 1
                break
        else:
            truncs += 1
        elapsed = time.time() - start
        print(f"Episode {ep+1}/{episodes}: steps={t+1}, captured={any(terms.values())}, time={elapsed:.3f}s")

    print("\nSummary:")
    print(f"episodes={episodes}, captures={captures}, truncations={truncs}, mean_capture_steps={ (sum(times)/len(times)) if times else 'N/A' }")


if __name__ == '__main__':
    run(episodes=20)
