"""
Thin PettingZoo ParallelEnv wrapper over the V1 pursuit core.

Maps string agent keys (``predator_0`` … ``predator_7``) to the core's
integer indices and exposes Gymnasium ``Box`` spaces for observations and
actions.
"""

from __future__ import annotations

import functools
from typing import Any

import gymnasium
import numpy as np
from pettingzoo import ParallelEnv

from swarm_env.config import DRONE_COUNT, DRONE_SPEED
from swarm_env.environment import Environment, OBS_SIZE


class PursuitParallelEnv(ParallelEnv):
    """PettingZoo Parallel API for the V1 pursuit environment."""

    metadata = {"name": "pursuit_v1", "render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None, **env_kwargs: Any):
        super().__init__()
        self._env = Environment(**env_kwargs)
        self.render_mode = render_mode

        self.possible_agents = [f"predator_{i}" for i in range(DRONE_COUNT)]
        self.agents = list(self.possible_agents)

        self._agent_to_idx = {a: i for i, a in enumerate(self.possible_agents)}
        self._idx_to_agent = {i: a for a, i in self._agent_to_idx.items()}

    # ── spaces ────────────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(
            low=-DRONE_SPEED, high=DRONE_SPEED, shape=(2,), dtype=np.float32,
        )

    # ── reset / step ──────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.agents = list(self.possible_agents)
        obs_int, infos_int = self._env.reset(seed=seed)
        obs = {self._idx_to_agent[i]: v for i, v in obs_int.items()}
        infos = {a: infos_int for a in self.agents}
        return obs, infos

    def step(
        self, actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        int_actions = {
            self._agent_to_idx[a]: (float(v[0]), float(v[1]))
            for a, v in actions.items()
        }
        obs_int, rew_int, term_int, trunc_int, infos_int = self._env.step(int_actions)

        obs = {self._idx_to_agent[i]: v for i, v in obs_int.items()}
        rew = {self._idx_to_agent[i]: v for i, v in rew_int.items()}
        term = {self._idx_to_agent[i]: v for i, v in term_int.items()}
        trunc = {self._idx_to_agent[i]: v for i, v in trunc_int.items()}
        infos = {a: infos_int for a in self.agents}

        # remove terminated / truncated agents from self.agents
        self.agents = [
            a for a in self.agents
            if not term.get(a, False) and not trunc.get(a, False)
        ]

        return obs, rew, term, trunc, infos

    # ── render ────────────────────────────────────────────────────────────

    def render(self) -> None:
        pass  # rendering handled by main.py pygame loop

    def close(self) -> None:
        pass
