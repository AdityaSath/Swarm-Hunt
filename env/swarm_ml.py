"""Shared MATD3 construction and behavior-cloning checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch
from agilerl.algorithms import MATD3

from swarm_env.config import DRONE_COUNT, DRONE_SPEED
from swarm_env.environment import OBS_SIZE

AGENT_IDS = [f"predator_{i}" for i in range(DRONE_COUNT)]
NET_CONFIG: dict[str, Any] = {
    "encoder_config": {"hidden_size": [128, 128]},
}


def observation_spaces() -> list[gymnasium.spaces.Box]:
    return [
        gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )
        for _ in AGENT_IDS
    ]


def action_spaces() -> list[gymnasium.spaces.Box]:
    return [
        gymnasium.spaces.Box(
            low=-DRONE_SPEED,
            high=DRONE_SPEED,
            shape=(2,),
            dtype=np.float32,
        )
        for _ in AGENT_IDS
    ]


def build_matd3(
    device: str | torch.device = "cpu",
    vect_noise_dim: int = 1,
    **overrides: Any,
) -> MATD3:
    """Build the architecture shared by BC, training, evaluation, and demo."""
    kwargs: dict[str, Any] = {
        "observation_spaces": observation_spaces(),
        "action_spaces": action_spaces(),
        "agent_ids": AGENT_IDS,
        "O_U_noise": False,
        "expl_noise": 0.10,
        "vect_noise_dim": vect_noise_dim,
        "net_config": NET_CONFIG,
        "batch_size": 256,
        "lr_actor": 3e-4,
        "lr_critic": 3e-4,
        "learn_step": 100,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_freq": 2,
        "device": str(device),
    }
    kwargs.update(overrides)
    return MATD3(**kwargs)


def save_bc_actor(agent: MATD3, path: str | Path, metadata: dict[str, Any]) -> None:
    """Save only policy weights, independent of critics and replay state."""
    actor_states = {
        network_id: {
            key: value.detach().cpu()
            for key, value in actor.state_dict().items()
        }
        for network_id, actor in agent.actors.items()
    }
    payload = {
        "format": "swarm_hunt_bc_v1",
        "obs_size": OBS_SIZE,
        "drone_count": DRONE_COUNT,
        "net_config": NET_CONFIG,
        "actor_state_dicts": actor_states,
        "metadata": metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_bc_actors(agent: MATD3, path: str | Path) -> dict[str, Any]:
    """Load BC policies into live and target actors of a MATD3 agent."""
    payload = torch.load(Path(path), map_location=agent.device, weights_only=True)
    if payload.get("format") != "swarm_hunt_bc_v1":
        raise ValueError(f"Unsupported behavior-cloning checkpoint: {path}")
    if payload.get("obs_size") != OBS_SIZE:
        raise ValueError(
            f"BC checkpoint observation size {payload.get('obs_size')} != {OBS_SIZE}"
        )

    actor_states = payload["actor_state_dicts"]
    if set(actor_states) != set(agent.actors.keys()):
        raise ValueError(
            f"BC actor groups {sorted(actor_states)} do not match "
            f"MATD3 groups {sorted(agent.actors.keys())}"
        )
    for network_id, state in actor_states.items():
        agent.actors[network_id].load_state_dict(state)
        agent.actor_targets[network_id].load_state_dict(state)
    return dict(payload.get("metadata", {}))
