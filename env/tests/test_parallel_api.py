"""Smoke test: PettingZoo parallel_api_test + one random episode."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pettingzoo.test import parallel_api_test
from swarm_env.parallel_env import PursuitParallelEnv


def test_parallel_api():
    env = PursuitParallelEnv(seed=42)
    parallel_api_test(env, num_cycles=50)
    print("PASS: parallel_api_test (50 cycles)")


def test_random_episode():
    env = PursuitParallelEnv(seed=0)
    obs, _ = env.reset(seed=0)
    assert len(obs) == 8
    total_steps = 0
    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        env.step(actions)
        total_steps += 1
        if total_steps > 1850:  # safety guard > 30 s timeout (1800 @ 60 FPS)
            break
    print(f"PASS: random episode ran {total_steps} steps")
    env.close()


def test_curriculum_env_options():
    env = PursuitParallelEnv(
        seed=0,
        enable_obstacles=False,
        always_visible=True,
        capture_hold_seconds=0.5,
    )
    obs, _ = env.reset(seed=0)
    assert len(env._env.obstacles) == 0
    assert all(agent_obs[4] == 1.0 for agent_obs in obs.values())
    env.close()


if __name__ == "__main__":
    test_parallel_api()
    test_random_episode()
    test_curriculum_env_options()
    print("\nAll PettingZoo tests passed.")
