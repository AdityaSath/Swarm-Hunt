"""Collect scripted formation-controller demonstrations into an HDF5 dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py
import numpy as np

from swarm_env.config import DRONE_COUNT, DRONE_SPEED
from swarm_env.environment import Environment, OBS_SIZE
from swarm_env.formation_controller import FormationController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect expert capture trajectories")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output", default="data/expert_capture.h5")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--min-prey-speed", type=float, default=0.25)
    parser.add_argument("--max-prey-speed", type=float, default=1.0)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--no-obstacles", action="store_true")
    parser.add_argument(
        "--include-timeouts",
        action="store_true",
        help="Keep failed expert episodes (successful captures only by default)",
    )
    return parser.parse_args()


class ExpertDatasetWriter:
    """Append episode-sized batches without keeping the full dataset in RAM."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")
        self.observations = self.file.create_dataset(
            "observations",
            shape=(0, OBS_SIZE),
            maxshape=(None, OBS_SIZE),
            chunks=(4096, OBS_SIZE),
            compression="gzip",
            dtype=np.float32,
        )
        self.actions = self.file.create_dataset(
            "actions",
            shape=(0, 2),
            maxshape=(None, 2),
            chunks=(4096, 2),
            compression="gzip",
            dtype=np.float32,
        )
        self.agent_indices = self.file.create_dataset(
            "agent_indices",
            shape=(0,),
            maxshape=(None,),
            chunks=(4096,),
            compression="gzip",
            dtype=np.int16,
        )
        self.episode_indices = self.file.create_dataset(
            "episode_indices",
            shape=(0,),
            maxshape=(None,),
            chunks=(4096,),
            compression="gzip",
            dtype=np.int32,
        )

    def append(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        agent_indices: np.ndarray,
        episode_indices: np.ndarray,
    ) -> None:
        start = len(self.observations)
        end = start + len(observations)
        for dataset in (
            self.observations,
            self.actions,
            self.agent_indices,
            self.episode_indices,
        ):
            dataset.resize(end, axis=0)
        self.observations[start:end] = observations
        self.actions[start:end] = actions
        self.agent_indices[start:end] = agent_indices
        self.episode_indices[start:end] = episode_indices

    def close(self) -> None:
        self.file.close()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.min_prey_speed > args.max_prey_speed:
        raise ValueError("--min-prey-speed cannot exceed --max-prey-speed")

    rng = random.Random(args.seed)
    writer = ExpertDatasetWriter(args.output)
    writer.file.attrs.update(
        {
            "format": "swarm_hunt_expert_v1",
            "obs_size": OBS_SIZE,
            "drone_count": DRONE_COUNT,
            "action_scale": DRONE_SPEED,
            "base_seed": args.seed,
            "action_repeat": max(1, args.action_repeat),
        }
    )

    captures = 0
    saved_episodes = 0
    total_decisions = 0
    try:
        for episode in range(args.episodes):
            prey_speed = rng.uniform(args.min_prey_speed, args.max_prey_speed)
            env = Environment(
                seed=args.seed + episode,
                prey_speed_factor=prey_speed,
                obstacles_enabled=not args.no_obstacles,
            )
            controller = FormationController()
            observations, _ = env.reset(seed=args.seed + episode)
            controller.reset(env)

            episode_obs: list[np.ndarray] = []
            episode_actions: list[np.ndarray] = []
            episode_agents: list[int] = []
            captured = False
            done = False

            while not done:
                actions = controller.actions(env)
                for agent_idx in range(env.num_agents):
                    episode_obs.append(observations[agent_idx])
                    episode_actions.append(
                        np.asarray(actions[agent_idx], dtype=np.float32) / DRONE_SPEED
                    )
                    episode_agents.append(agent_idx)

                for _ in range(max(1, args.action_repeat)):
                    observations, _, terminations, truncations, _ = env.step(actions)
                    captured = any(terminations.values())
                    done = captured or any(truncations.values())
                    if done:
                        break

            captures += int(captured)
            if captured or args.include_timeouts:
                count = len(episode_obs)
                writer.append(
                    np.asarray(episode_obs, dtype=np.float32),
                    np.asarray(episode_actions, dtype=np.float32),
                    np.asarray(episode_agents, dtype=np.int16),
                    np.full(count, episode, dtype=np.int32),
                )
                total_decisions += count // DRONE_COUNT
                saved_episodes += 1

            if (episode + 1) % 10 == 0 or episode + 1 == args.episodes:
                print(
                    f"episodes={episode + 1}/{args.episodes}  "
                    f"captures={captures}  saved={saved_episodes}  "
                    f"samples={len(writer.observations):,}"
                )
    finally:
        writer.file.attrs["episodes"] = args.episodes
        writer.file.attrs["captures"] = captures
        writer.file.attrs["saved_episodes"] = saved_episodes
        writer.file.attrs["decisions"] = total_decisions
        writer.close()

    print(f"Saved expert dataset to {args.output}")


if __name__ == "__main__":
    main()
