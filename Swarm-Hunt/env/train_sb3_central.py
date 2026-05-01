"""Simple SB3 training script for centralized control of all predators.

Usage:
  source .venv/bin/activate
  pip install stable-baselines3[extra] tensorboard
  python env/train_sb3_central.py --timesteps 50000 --n-envs 4

This script trains a single PPO policy that outputs a concatenated velocity
vector for all predators.
"""
import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

from gym_centralized import CentralizedSwarmGym, HybridSwarmGym


def _configure_matplotlib_cache_dir() -> None:
    """Ensure Matplotlib has a writable config/cache directory.

    Stable-Baselines3 imports Matplotlib for optional logging helpers. On some
    setups (or in sandboxed environments) the default user config dir may be
    unwritable, which can make the first import very slow or error-prone.
    """

    if os.environ.get("MPLCONFIGDIR"):
        return

    repo_root = Path(__file__).resolve().parents[1]
    mpl_dir = repo_root / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")


class ScriptedWarmStartWrapper(gym.Env):
    """Wrap a CentralizedSwarmGym and optionally replace agent actions with
    the environment's scripted actions for the first N steps (warm start).

    The underlying centralized env instance is available as `self.env`.
    """

    def __init__(self, env: CentralizedSwarmGym, warm_steps: int = 0, warm_prob: float = 0.5):
        self.env = env
        self.warm_steps = int(warm_steps)
        self.warm_prob = float(warm_prob)
        self._step_count = 0

        # expose spaces
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray):
        # decide whether to use scripted action
        use_scripted = (self._step_count < self.warm_steps) and (np.random.rand() < self.warm_prob)
        if use_scripted:
            # scripted_actions returns a dict {agent_idx: (vx, vy)}
            scripted = self.env.env.scripted_actions()
            # convert to centralized action vector
            act_vec = np.zeros(self.action_space.shape, dtype=np.float32)
            for i in range(self.env.drone_count):
                vx, vy = scripted.get(i, (0.0, 0.0))
                act_vec[2 * i] = float(vx)
                act_vec[2 * i + 1] = float(vy)
            obs, reward, terminated, truncated, info = self.env.step(act_vec)
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)

        self._step_count += 1
        return obs, reward, terminated, truncated, info


def make_env_fn(seed: int = 0, warm_steps: int = 0, warm_prob: float = 0.0, hybrid: bool = False):
    def _fn():
        env = HybridSwarmGym() if hybrid else CentralizedSwarmGym()
        # optionally wrap for warm-start scripted actions
        if warm_steps > 0 and warm_prob > 0.0:
            env = ScriptedWarmStartWrapper(env, warm_steps=warm_steps, warm_prob=warm_prob)
        # Gymnasium-style seeding: call reset with seed when creating the env
        try:
            env.reset(seed=seed)
        except TypeError:
            # older envs may not accept seed in reset
            pass
        return env

    return _fn


def main():
    # Configure Matplotlib before importing stable-baselines3. Also defer SB3
    # imports until inside main() so SubprocVecEnv workers don't pay the import
    # cost or trigger Matplotlib font-cache work.
    _configure_matplotlib_cache_dir()
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--logdir", type=str, default="./sb3_logs")
    parser.add_argument("--warm-start-steps", type=int, default=0,
                        help="Number of env steps to allow scripted-action warm start")
    parser.add_argument("--warm-start-prob", type=float, default=0.0,
                        help="Probability of using scripted action during warm start")
    parser.add_argument("--hybrid-actions", action="store_true",
                        help="Train high-level search/flank intent instead of raw velocities")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    if args.n_envs > 1:
        env_fns = [
            make_env_fn(
                i,
                warm_steps=args.warm_start_steps,
                warm_prob=args.warm_start_prob,
                hybrid=args.hybrid_actions,
            )
            for i in range(args.n_envs)
        ]
        # On some macOS/sandboxed setups, the default start method ("forkserver")
        # can fail with permission errors; "spawn" is slower but more robust.
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv([
            make_env_fn(
                0,
                warm_steps=args.warm_start_steps,
                warm_prob=args.warm_start_prob,
                hybrid=args.hybrid_actions,
            )
        ])

    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=args.logdir)
    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(args.logdir, "ppo_central"))


if __name__ == "__main__":
    main()
