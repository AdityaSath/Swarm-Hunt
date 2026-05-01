import numpy as np
import gymnasium as gym
from gymnasium import spaces

from swarm_env.environment import Environment, OBS_SIZE
from swarm_env.config import DRONE_COUNT, DRONE_SPEED


class CentralizedSwarmGym(gym.Env):
    """Gym wrapper that exposes a centralized observation/action space.

    Observation: concatenation of per-predator observations (OBS_SIZE * N)
    Action: concatenation of per-predator desired velocities (2 * N)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, drone_count: int = DRONE_COUNT, width: int | None = None, height: int | None = None):
        super().__init__()
        self.drone_count = int(drone_count)
        # underlying env
        self.env = Environment(drone_count=self.drone_count)

        obs_dim = OBS_SIZE * self.drone_count
        act_dim = 2 * self.drone_count

        # unbounded observation (values already normalized in env by WORLD_SCALE)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # action: 2D desired velocity per agent (clipped by DRONE_SPEED in Drone.set_desired_velocity)
        self.action_space = spaces.Box(low=-DRONE_SPEED, high=DRONE_SPEED, shape=(act_dim,), dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs_dict, info = self.env.reset(seed=seed)
        return self._concat_obs(obs_dict), info or {}

    def step(self, action: np.ndarray):
        # split action vector into per-agent (vx, vy)
        assert action.shape[0] == 2 * self.drone_count
        actions = {i: (float(action[2 * i]), float(action[2 * i + 1])) for i in range(self.drone_count)}
        obs_dict, rewards, terminations, truncations, info = self.env.step(actions)
        # centralized reward: sum team reward
        reward = float(sum(rewards.values()))
        terminated = any(terminations.values())
        truncated = any(truncations.values())
        obs = self._concat_obs(obs_dict)
        # Gymnasium-style step return: obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        # For interactive rendering use env.render(screen) from main.py loop.
        raise NotImplementedError("Use the demo `env/main.py` for rendering.")

    def close(self):
        pass

    def _concat_obs(self, obs_dict: dict[int, np.ndarray]) -> np.ndarray:
        # ensure ordering by agent index
        obs_list = [obs_dict[i].astype(np.float32) for i in range(self.drone_count)]
        return np.concatenate(obs_list, axis=0)


class HybridSwarmGym(CentralizedSwarmGym):
    """Centralized wrapper where RL chooses high-level intent, not velocity.

    Action layout per predator: ``target_x, target_y, flank_angle`` in [-1, 1].
    The environment's hybrid controller converts this into clipped velocities.
    """

    def __init__(self, drone_count: int = DRONE_COUNT, width: int | None = None, height: int | None = None):
        super().__init__(drone_count=drone_count, width=width, height=height)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3 * self.drone_count,),
            dtype=np.float32,
        )

    def step(self, action: np.ndarray):
        assert action.shape[0] == 3 * self.drone_count
        actions = self.env.hybrid_actions(action)
        obs_dict, rewards, terminations, truncations, info = self.env.step(actions)
        reward = float(sum(rewards.values()))
        terminated = any(terminations.values())
        truncated = any(truncations.values())
        obs = self._concat_obs(obs_dict)
        return obs, reward, terminated, truncated, info
