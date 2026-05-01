"""Headless evaluator for the centralized PPO policy.

Runs several episodes without rendering and prints per-episode metrics.
Useful for debugging whether a saved model produces valid actions and
whether the environment terminates/captures as expected.

Usage:
  .venv/bin/python env/eval_headless.py --model ./sb3_logs_smoke/ppo_central.zip --episodes 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np

from gym_centralized import CentralizedSwarmGym


def _configure_matplotlib_cache_dir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return

    repo_root = Path(__file__).resolve().parents[1]
    mpl_dir = repo_root / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")


def run_headless(model_path: str, episodes: int = 5, max_steps: int = 1000, deterministic: bool = True):
    _configure_matplotlib_cache_dir()
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    env = CentralizedSwarmGym()

    results = []
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        t0 = time.time()
        done = False
        while steps < max_steps and not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)

        elapsed = time.time() - t0
        results.append({"steps": steps, "reward": total_reward, "terminated": terminated, "truncated": truncated, "time": elapsed})
        print(f"Episode {ep+1}: steps={steps}, reward={total_reward:.3f}, terminated={terminated}, truncated={truncated}, time={elapsed:.3f}s")

    print("\nSummary:")
    print(f"episodes={episodes}, mean_steps={mean(r['steps'] for r in results):.1f}, mean_reward={mean(r['reward'] for r in results):.3f}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_headless(args.model, episodes=args.episodes, max_steps=args.max_steps, deterministic=args.deterministic)


if __name__ == "__main__":
    main()
