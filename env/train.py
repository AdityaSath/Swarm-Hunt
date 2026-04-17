"""
MATD3 training script for the V1 pursuit environment using AgileRL.

Supports curriculum learning with automatic prey speed progression.

Usage:
    python train.py                          # curriculum ON by default
    python train.py --no-curriculum          # full difficulty from start
    python train.py --max-steps 5000000      # longer budget
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from swarm_env.parallel_env import PursuitParallelEnv

CURRICULUM_STAGES = [
    {"prey_speed_factor": 0.5, "label": "Stage 1 (prey 0.5x)"},
    {"prey_speed_factor": 0.75, "label": "Stage 2 (prey 0.75x)"},
    {"prey_speed_factor": 1.0, "label": "Stage 3 (prey 1.0x — full)"},
]
CURRICULUM_ADVANCE_RATE = 0.30
CURRICULUM_WINDOW = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MATD3 on pursuit_v1")
    p.add_argument("--max-steps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--action-repeat", type=int, default=4,
                   help="Env sub-steps per agent decision (frame skip)")
    p.add_argument("--pop-size", type=int, default=4)
    p.add_argument("--evo-steps", type=int, default=4000,
                   help="Steps between evolutionary tournaments")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--no-curriculum", action="store_true",
                   help="Disable curriculum (use full prey speed from start)")
    p.add_argument("--save-dir", type=str, default="./models/MATD3")
    p.add_argument("--log-dir", type=str, default="./runs")
    p.add_argument("--save-every", type=int, default=5,
                   help="Save checkpoint every N generations")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_env(action_repeat: int, prey_speed_factor: float):
    """Factory that returns a new PursuitParallelEnv instance."""
    def _thunk():
        return PursuitParallelEnv(
            action_repeat=action_repeat,
            prey_speed_factor=prey_speed_factor,
        )
    return _thunk


def build_vec_env(num_envs: int, action_repeat: int, prey_speed_factor: float):
    env = AsyncPettingZooVecEnv(
        [make_env(action_repeat, prey_speed_factor) for _ in range(num_envs)]
    )
    env.reset()
    return env


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Device: {device}")

    # ── curriculum setup ──────────────────────────────────────────────────
    if args.no_curriculum:
        stages = [{"prey_speed_factor": 1.0, "label": "Full difficulty"}]
    else:
        stages = list(CURRICULUM_STAGES)

    current_stage = 0
    prey_speed_factor = stages[current_stage]["prey_speed_factor"]

    # ── environment ───────────────────────────────────────────────────────
    env = build_vec_env(args.num_envs, args.action_repeat, prey_speed_factor)

    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]

    # ── hyperparameters ───────────────────────────────────────────────────
    NET_CONFIG = {
        "encoder_config": {
            "hidden_size": [128, 128],
        }
    }

    INIT_HP = {
        "POPULATION_SIZE": args.pop_size,
        "ALGO": "MATD3",
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 256,
        "O_U_NOISE": True,
        "EXPL_NOISE": 0.15,
        "MEAN_NOISE": 0.0,
        "THETA": 0.15,
        "DT": 0.01,
        "LR_ACTOR": 3e-4,
        "LR_CRITIC": 3e-4,
        "GAMMA": 0.99,
        "MEMORY_SIZE": 500_000,
        "LEARN_STEP": 100,
        "TAU": 0.005,
        "POLICY_FREQ": 2,
        "N_AGENTS": env.num_agents,
        "AGENT_IDS": env.agents,
    }

    hp_config = HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=1e-2),
        lr_critic=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=64, max=512, dtype=int),
        learn_step=RLParameter(
            min=50, max=300, dtype=int, grow_factor=1.5, shrink_factor=0.75
        ),
    )

    # ── population ────────────────────────────────────────────────────────
    pop = create_population(
        algo=INIT_HP["ALGO"],
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        observation_space=observation_spaces,
        action_space=action_spaces,
        hp_config=hp_config,
        population_size=INIT_HP["POPULATION_SIZE"],
        num_envs=args.num_envs,
        device=device,
    )

    # ── replay buffer ─────────────────────────────────────────────────────
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    # ── evolutionary HPO ──────────────────────────────────────────────────
    tournament = TournamentSelection(
        tournament_size=2,
        elitism=True,
        population_size=INIT_HP["POPULATION_SIZE"],
        eval_loop=1,
    )

    mutations = Mutations(
        no_mutation=0.4,
        architecture=0.1,
        new_layer_prob=0.1,
        parameters=0.2,
        activation=0,
        rl_hp=0.2,
        mutation_sd=0.1,
        rand_seed=args.seed,
        device=device,
    )

    # ── logging ───────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    elite = pop[0]
    total_steps = 0
    generation = 0
    episodes_completed = 0
    captures = 0
    timeouts = 0
    recent_outcomes: deque[bool] = deque(maxlen=CURRICULUM_WINDOW)
    t_start = time.time()

    # ── training loop ─────────────────────────────────────────────────────
    print(f"Training MATD3  |  max_steps={args.max_steps}  |  "
          f"num_envs={args.num_envs}  |  pop_size={args.pop_size}  |  "
          f"action_repeat={args.action_repeat}")
    print(f"Curriculum: {stages[current_stage]['label']}")
    pbar = trange(args.max_steps, unit="step")

    while np.less([agent.steps[-1] for agent in pop], args.max_steps).all():
        pop_episode_scores = []

        for agent in pop:
            state, info = env.reset()
            scores = np.zeros(args.num_envs)
            completed_episode_scores = []
            steps = 0

            for idx_step in range(args.evo_steps // args.num_envs):
                cont_actions, _ = agent.get_action(state, infos=info)
                action = cont_actions

                next_state, reward, termination, truncation, info = env.step(action)

                scores += np.sum(
                    np.array(list(reward.values())).transpose(), axis=-1
                )
                total_steps += args.num_envs
                steps += args.num_envs

                done = {
                    agent_id: np.logical_or(termination[agent_id], truncation[agent_id])
                    for agent_id in termination
                }

                memory.save_to_memory(
                    state,
                    cont_actions,
                    reward,
                    next_state,
                    done,
                    is_vectorised=True,
                )

                # ── learning ──────────────────────────────────────────
                if agent.learn_step > args.num_envs:
                    learn_step = agent.learn_step // args.num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                    ):
                        experiences = memory.sample(agent.batch_size)
                        agent.learn(experiences)
                elif len(memory) >= agent.batch_size:
                    for _ in range(args.num_envs // agent.learn_step):
                        experiences = memory.sample(agent.batch_size)
                        agent.learn(experiences)

                state = next_state

                # ── episode bookkeeping ───────────────────────────────
                reset_noise_indices = []
                term_array = np.array(list(termination.values())).transpose()
                trunc_array = np.array(list(truncation.values())).transpose()
                for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                    if np.any(d) or np.any(t):
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)
                        episodes_completed += 1

                        was_capture = bool(np.any(d))
                        recent_outcomes.append(was_capture)
                        if was_capture:
                            captures += 1
                        else:
                            timeouts += 1

                agent.reset_action_noise(reset_noise_indices)

            pbar.update(args.evo_steps // len(pop))
            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # ── evaluation (with explicit step cap to prevent hangs) ──────────
        fitnesses = [
            agent.test(env, swap_channels=False, max_steps=500, loop=1)
            for agent in pop
        ]
        mean_scores = [
            np.mean(ep) if len(ep) > 0 else 0.0
            for ep in pop_episode_scores
        ]

        # ── curriculum advancement ────────────────────────────────────────
        recent_capture_rate = (
            sum(recent_outcomes) / len(recent_outcomes)
            if len(recent_outcomes) >= CURRICULUM_WINDOW
            else 0.0
        )

        if (
            current_stage < len(stages) - 1
            and len(recent_outcomes) >= CURRICULUM_WINDOW
            and recent_capture_rate >= CURRICULUM_ADVANCE_RATE
        ):
            current_stage += 1
            prey_speed_factor = stages[current_stage]["prey_speed_factor"]
            print(f"\n*** CURRICULUM ADVANCE -> {stages[current_stage]['label']} "
                  f"(capture rate {recent_capture_rate:.0%}) ***")
            env.close()
            env = build_vec_env(args.num_envs, args.action_repeat, prey_speed_factor)
            recent_outcomes.clear()

        # ── TensorBoard logging ──────────────────────────────────────────
        writer.add_scalar("train/mean_score", np.mean(mean_scores), total_steps)
        writer.add_scalar("train/elite_fitness", max(fitnesses), total_steps)
        writer.add_scalar("train/episodes", episodes_completed, total_steps)
        writer.add_scalar("train/capture_rate_recent", recent_capture_rate, total_steps)
        writer.add_scalar("train/captures_total", captures, total_steps)
        writer.add_scalar("train/timeouts_total", timeouts, total_steps)
        writer.add_scalar("train/curriculum_stage", current_stage, total_steps)
        writer.add_scalar("train/prey_speed_factor", prey_speed_factor, total_steps)
        writer.add_scalar(
            "train/wall_time_min", (time.time() - t_start) / 60, total_steps
        )
        writer.add_scalar("train/replay_buffer_size", len(memory), total_steps)

        elapsed = time.time() - t_start
        print(
            f"\n--- Steps {total_steps:,} | "
            f"Ep {episodes_completed} | "
            f"Cap {captures} | TO {timeouts} | "
            f"Rate {recent_capture_rate:.0%} | "
            f"{stages[current_stage]['label']} | "
            f"Scores {[f'{s:.1f}' for s in mean_scores]} | "
            f"Fit {[f'{f:.1f}' for f in fitnesses]} | "
            f"{elapsed/60:.1f}m ---"
        )

        # ── evolution ─────────────────────────────────────────────────────
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

        for agent in pop:
            agent.steps.append(agent.steps[-1])

        # ── periodic checkpoint ───────────────────────────────────────────
        generation += 1
        if generation % args.save_every == 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = os.path.join(args.save_dir, f"MATD3_gen{generation}_{ts}.pt")
            elite.save_checkpoint(ckpt_path)
            print(f"  [checkpoint] gen {generation} -> {ckpt_path}")

    # ── final save ────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, f"MATD3_final_{ts}.pt")
    elite.save_checkpoint(save_path)
    print(f"\nElite agent saved to {save_path}")

    pbar.close()
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
