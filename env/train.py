"""Fine-tune a role-aware MATD3 swarm, optionally from behavior cloning.

Compatible with AgileRL 2.14's TensorDict replay-buffer API.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from agilerl.components import ReplayBuffer
from agilerl.components.data import MultiAgentTransition, transition_to_tensordict
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from swarm_env.config import DRONE_COUNT, FPS
from swarm_env.environment import Environment
from swarm_env.parallel_env import PursuitParallelEnv
from swarm_ml import AGENT_IDS, build_matd3, load_bc_actors


CURRICULUM_STAGES: list[dict[str, Any]] = [
    {
        "label": "Stage 0 - stationary prey, open arena",
        "prey_speed_factor": 0.0,
        "obstacles_enabled": False,
        "capture_hold_seconds": 0.5,
    },
    {
        "label": "Stage 1 - slow prey, open arena",
        "prey_speed_factor": 0.25,
        "obstacles_enabled": False,
        "capture_hold_seconds": 1.0,
    },
    {
        "label": "Stage 2 - moving prey with obstacles",
        "prey_speed_factor": 0.5,
        "obstacles_enabled": True,
        "capture_hold_seconds": 1.0,
    },
    {
        "label": "Stage 3 - fast prey with obstacles",
        "prey_speed_factor": 0.75,
        "obstacles_enabled": True,
        "capture_hold_seconds": 1.5,
    },
    {
        "label": "Stage 4 - full task",
        "prey_speed_factor": 1.0,
        "obstacles_enabled": True,
        "capture_hold_seconds": 2.0,
    },
]
CURRICULUM_WINDOW = 100
CURRICULUM_ADVANCE_RATE = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train role-aware MATD3 pursuit")
    parser.add_argument("--max-steps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=4_000,
        help="Vector transitions between evaluation/checkpoints",
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=str,
        default=None,
        help="Actor checkpoint produced by pretrain_bc.py",
    )
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-dir", default="models/MATD3")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def make_env(action_repeat: int, stage: dict[str, Any], seed: int):
    def _thunk() -> PursuitParallelEnv:
        return PursuitParallelEnv(
            action_repeat=action_repeat,
            seed=seed,
            prey_speed_factor=stage["prey_speed_factor"],
            obstacles_enabled=stage["obstacles_enabled"],
            capture_hold_steps=max(1, round(stage["capture_hold_seconds"] * FPS)),
        )

    return _thunk


def build_vec_env(
    num_envs: int,
    action_repeat: int,
    stage: dict[str, Any],
    base_seed: int,
) -> AsyncPettingZooVecEnv:
    return AsyncPettingZooVecEnv(
        [
            make_env(action_repeat, stage, base_seed + env_idx)
            for env_idx in range(num_envs)
        ]
    )


def combine_done(
    termination: dict[str, np.ndarray],
    truncation: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Replay must stop bootstrapping at both captures and time limits."""
    return {
        agent_id: np.logical_or(
            np.asarray(termination[agent_id], dtype=bool),
            np.asarray(truncation[agent_id], dtype=bool),
        )
        for agent_id in termination
    }


def evaluate_capture_rate(
    agent,
    stage: dict[str, Any],
    episodes: int,
    seed_start: int,
    action_repeat: int,
) -> tuple[float, float]:
    """Return capture rate and mean core-physics steps on held-out seeds."""
    if episodes <= 0:
        return 0.0, 0.0
    previous_mode = agent.training
    agent.set_training_mode(False)
    captures = 0
    episode_steps: list[int] = []

    try:
        for episode in range(episodes):
            env = Environment(
                seed=seed_start + episode,
                prey_speed_factor=stage["prey_speed_factor"],
                obstacles_enabled=stage["obstacles_enabled"],
                capture_hold_steps=max(
                    1, round(stage["capture_hold_seconds"] * FPS)
                ),
            )
            obs_int, _ = env.reset(seed=seed_start + episode)
            done = False
            while not done:
                obs = {AGENT_IDS[i]: value for i, value in obs_int.items()}
                processed_actions, _ = agent.get_action(obs)
                actions = {
                    i: tuple(
                        float(component)
                        for component in np.asarray(
                            processed_actions[AGENT_IDS[i]]
                        ).reshape(-1)[:2]
                    )
                    for i in range(DRONE_COUNT)
                }
                for _ in range(max(1, action_repeat)):
                    obs_int, _, terms, truncs, _ = env.step(actions)
                    if any(terms.values()):
                        captures += 1
                        done = True
                    elif any(truncs.values()):
                        done = True
                    if done:
                        break
            episode_steps.append(env._step_count)
    finally:
        agent.set_training_mode(previous_mode)

    return captures / episodes, float(np.mean(episode_steps))


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    stages = [CURRICULUM_STAGES[-1]] if args.no_curriculum else list(CURRICULUM_STAGES)
    stage_idx = 0
    stage = stages[stage_idx]
    env = build_vec_env(args.num_envs, args.action_repeat, stage, args.seed)
    obs, info = env.reset()

    agent = build_matd3(device=device, vect_noise_dim=args.num_envs)
    if args.bc_checkpoint:
        metadata = load_bc_actors(agent, args.bc_checkpoint)
        print(f"Loaded BC actor: {args.bc_checkpoint}  metadata={metadata}")

    memory = ReplayBuffer(max_size=500_000, device=device)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    recent_outcomes: deque[bool] = deque(maxlen=CURRICULUM_WINDOW)
    total_steps = 0
    generation = 0
    best_stage_rate = -1.0
    episodes_completed = 0
    captures = 0
    timeouts = 0
    started = time.time()

    print(
        f"Device: {device}  |  num_envs={args.num_envs}  |  "
        f"action_repeat={args.action_repeat}"
    )
    print(f"Curriculum: {stage['label']}")
    progress = tqdm(total=args.max_steps, unit="step")

    try:
        while total_steps < args.max_steps:
            chunk_steps = min(args.rollout_steps, args.max_steps - total_steps)
            vector_iterations = max(1, chunk_steps // args.num_envs)
            agent.set_training_mode(True)

            for iteration in range(vector_iterations):
                processed_actions, raw_actions = agent.get_action(obs, infos=info)
                next_obs, reward, termination, truncation, info = env.step(
                    processed_actions
                )
                dones = combine_done(termination, truncation)
                transition = transition_to_tensordict(
                    MultiAgentTransition(
                        obs=obs,
                        action=raw_actions,
                        reward=reward,
                        next_obs=next_obs,
                        done=dones,
                    )
                )
                transition.batch_size = torch.Size([args.num_envs])
                memory.add(transition)

                if agent.learn_step > args.num_envs:
                    learn_every = max(1, agent.learn_step // args.num_envs)
                    if iteration % learn_every == 0 and len(memory) >= agent.batch_size:
                        agent.learn(memory.sample(agent.batch_size))
                elif len(memory) >= agent.batch_size:
                    for _ in range(max(1, args.num_envs // agent.learn_step)):
                        agent.learn(memory.sample(agent.batch_size))

                reset_noise_indices: list[int] = []
                done_rows = np.column_stack(
                    [np.asarray(dones[agent_id], dtype=bool) for agent_id in AGENT_IDS]
                )
                captured_rows = np.column_stack(
                    [
                        np.asarray(termination[agent_id], dtype=bool)
                        for agent_id in AGENT_IDS
                    ]
                )
                for env_idx, row in enumerate(done_rows):
                    if bool(np.all(row)):
                        was_capture = bool(np.any(captured_rows[env_idx]))
                        recent_outcomes.append(was_capture)
                        episodes_completed += 1
                        captures += int(was_capture)
                        timeouts += int(not was_capture)
                        reset_noise_indices.append(env_idx)
                agent.reset_action_noise(reset_noise_indices)

                obs = next_obs
                total_steps += args.num_envs
                agent.steps += args.num_envs
                progress.update(args.num_envs)
                if total_steps >= args.max_steps:
                    break

            generation += 1
            eval_rate, eval_mean_steps = evaluate_capture_rate(
                agent,
                stage,
                args.eval_episodes,
                seed_start=args.seed + 1_000_000 + generation * args.eval_episodes,
                action_repeat=args.action_repeat,
            )
            recent_rate = (
                sum(recent_outcomes) / len(recent_outcomes)
                if recent_outcomes
                else 0.0
            )

            writer.add_scalar("train/capture_rate_recent", recent_rate, total_steps)
            writer.add_scalar("eval/capture_rate", eval_rate, total_steps)
            writer.add_scalar("eval/mean_core_steps", eval_mean_steps, total_steps)
            writer.add_scalar("train/curriculum_stage", stage_idx, total_steps)
            writer.add_scalar("train/replay_size", len(memory), total_steps)

            if eval_rate > best_stage_rate:
                best_stage_rate = eval_rate
                best_path = os.path.join(args.save_dir, "MATD3_best.pt")
                agent.save_checkpoint(best_path)
                print(f"\nSaved best checkpoint: {best_path} ({eval_rate:.0%})")

            elapsed_min = (time.time() - started) / 60.0
            print(
                f"\nsteps={total_steps:,}  episodes={episodes_completed}  "
                f"captures={captures}  timeouts={timeouts}  "
                f"train_rate={recent_rate:.0%}  eval_rate={eval_rate:.0%}  "
                f"stage={stage_idx}  elapsed={elapsed_min:.1f}m"
            )

            if (
                stage_idx < len(stages) - 1
                and len(recent_outcomes) == CURRICULUM_WINDOW
                and recent_rate >= CURRICULUM_ADVANCE_RATE
            ):
                stage_idx += 1
                stage = stages[stage_idx]
                best_stage_rate = -1.0
                recent_outcomes.clear()
                env.close()
                env = build_vec_env(
                    args.num_envs,
                    args.action_repeat,
                    stage,
                    args.seed + stage_idx * 100_000,
                )
                obs, info = env.reset()
                print(f"\n*** CURRICULUM ADVANCE: {stage['label']} ***")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(args.save_dir, f"MATD3_final_{timestamp}.pt")
        agent.save_checkpoint(final_path)
        print(f"Saved final checkpoint: {final_path}")
    finally:
        progress.close()
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
