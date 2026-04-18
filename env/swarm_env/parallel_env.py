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

from swarm_env.environment import Environment, OBS_SIZE
from swarm_env.policy_actions import action_to_velocity, make_policy_action_space


class PursuitParallelEnv(ParallelEnv):
    """PettingZoo Parallel API for the V1 pursuit environment.

    Parameters
    ----------
    render_mode : str | None
        Ignored during headless training; kept for PettingZoo API compat.
    action_repeat : int
        Number of core-env sub-steps per ``step()`` call.  Rewards are
        summed across repeats; observations and done flags come from the
        final sub-step.  Default ``1`` (no skip).
    **env_kwargs
        Forwarded to ``Environment.__init__`` (width, height, drone_count,
        dt, seed).
    """

    metadata = {"name": "pursuit_v1", "render_modes": ["human"]}

    def __init__(
        self,
        render_mode: str | None = None,
        action_repeat: int = 1,
        **env_kwargs: Any,
    ):
        super().__init__()
        self._env = Environment(**env_kwargs)
        self.render_mode = render_mode
        self._action_repeat = max(1, int(action_repeat))

        self.possible_agents = [f"predator_{i}" for i in range(self._env.num_agents)]
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
        return make_policy_action_space()

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
            self._agent_to_idx[a]: action_to_velocity(
                v,
                self.action_space(a),
            )
            for a, v in actions.items()
        }

        cumulative_rew: dict[int, float] = {i: 0.0 for i in range(self._env.num_agents)}

        for _ in range(self._action_repeat):
            obs_int, rew_int, term_int, trunc_int, infos_int = self._env.step(int_actions)
            for i, r in rew_int.items():
                cumulative_rew[i] += r
            if any(term_int.values()) or any(trunc_int.values()):
                break

        obs = {self._idx_to_agent[i]: v for i, v in obs_int.items()}
        rew = {self._idx_to_agent[i]: cumulative_rew[i] for i in range(self._env.num_agents)}
        term = {self._idx_to_agent[i]: v for i, v in term_int.items()}
        trunc = {self._idx_to_agent[i]: v for i, v in trunc_int.items()}
        infos = {a: infos_int for a in self.agents}

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
