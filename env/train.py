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

from swarm_env.capture import PreyTacticalState
from swarm_env.environment import Environment
from swarm_env.parallel_env import PursuitParallelEnv
from swarm_env.policy_actions import action_to_velocity

CURRICULUM_STAGES = [
    {
        "label": "Stage 1 (stationary visible prey, no obstacles, hold 0.25s)",
        "advance_rate": 0.60,
        "env_kwargs": {
            "prey_speed_factor": 0.0,
            "enable_obstacles": False,
            "always_visible": True,
            "capture_hold_seconds": 0.25,
        },
    },
    {
        "label": "Stage 2 (visible, no obstacles, prey 0.20x, hold 0.5s)",
        "advance_rate": 0.50,
        "env_kwargs": {
            "prey_speed_factor": 0.2,
            "enable_obstacles": False,
            "always_visible": True,
            "capture_hold_seconds": 0.5,
        },
    },
    {
        "label": "Stage 3 (visible, no obstacles, prey 0.35x, hold 1.0s)",
        "advance_rate": 0.40,
        "env_kwargs": {
            "prey_speed_factor": 0.35,
            "enable_obstacles": False,
            "always_visible": True,
            "capture_hold_seconds": 1.0,
        },
    },
    {
        "label": "Stage 4 (no obstacles, prey 0.5x, hold 1.5s)",
        "advance_rate": 0.30,
        "env_kwargs": {
            "prey_speed_factor": 0.5,
            "enable_obstacles": False,
            "always_visible": False,
            "capture_hold_seconds": 1.5,
        },
    },
    {
        "label": "Stage 5 (obstacles on, prey 0.75x, hold 2.0s)",
        "advance_rate": 0.25,
        "env_kwargs": {
            "prey_speed_factor": 0.75,
            "enable_obstacles": True,
            "always_visible": False,
            "capture_hold_seconds": 2.0,
        },
    },
    {
        "label": "Stage 6 (full difficulty)",
        "advance_rate": 0.20,
        "env_kwargs": {
            "prey_speed_factor": 1.0,
            "enable_obstacles": True,
            "always_visible": False,
            "capture_hold_seconds": 2.0,
        },
    },
]
CURRICULUM_ADVANCE_RATE = 0.30
CURRICULUM_WINDOW = 50


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
    p.add_argument("--eval-episodes", type=int, default=8,
                   help="Fixed-seed evaluation episodes per generation")
    p.add_argument("--random-action-steps", type=int, default=10_000,
                   help="Per-agent env steps to collect random moving actions")
    p.add_argument("--learning-starts", type=int, default=10_000,
                   help="Per-agent env steps to collect before gradient updates")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_env(action_repeat: int, env_kwargs: dict[str, float | bool]):
    """Factory that returns a new PursuitParallelEnv instance."""
    def _thunk():
        return PursuitParallelEnv(
            action_repeat=action_repeat,
            **env_kwargs,
        )
    return _thunk


def build_vec_env(
    num_envs: int,
    action_repeat: int,
    env_kwargs: dict[str, float | bool],
):
    env = AsyncPettingZooVecEnv(
        [make_env(action_repeat, env_kwargs) for _ in range(num_envs)]
    )
    env.reset()
    return env


def sample_vectorized_random_actions(
    agent_ids: list[str],
    action_space_by_agent: dict[str, object],
    num_envs: int,
) -> dict[str, np.ndarray]:
    """Sample one batched random action per agent for AgileRL vector envs."""
    return {
        agent_id: np.stack(
            [action_space_by_agent[agent_id].sample() for _ in range(num_envs)]
        ).astype(np.float32)
        for agent_id in agent_ids
    }


def evaluate_policy(
    agent,
    action_repeat: int,
    env_kwargs: dict[str, float | bool],
    episodes: int,
    base_seed: int,
) -> dict[str, float]:
    action_space = _agent_action_space(agent)
    prev_training = agent.training
    agent.training = False
    captures = 0
    max_holds: list[int] = []
    max_combined: list[int] = []
    visible_fracs: list[float] = []
    visible_alignments: list[float] = []
    action_speeds: list[float] = []
    threatened_steps = 0
    total_steps = 0

    try:
        for ep in range(episodes):
            env = Environment(seed=base_seed + ep, **env_kwargs)
            repeat_left = 0
            cached_actions: dict[int, tuple[float, float]] | None = None
            ep_visible_steps = 0
            ep_steps = 0
            ep_max_hold = 0
            ep_max_combined = 0

            while True:
                if repeat_left <= 0:
                    obs_int = env._compute_observations()
                    obs = {
                        agent.agent_ids[i]: value
                        for i, value in obs_int.items()
                    }
                    cont_actions, _ = agent.get_action(obs)
                    cached_actions = {}
                    for i, agent_id in enumerate(agent.agent_ids):
                        action = np.asarray(cont_actions[agent_id]).reshape(-1)
                        vx, vy = action_to_velocity(action, action_space)
                        cached_actions[i] = (vx, vy)
                        action_speed = float(np.hypot(vx, vy))
                        action_speeds.append(action_speed)
                        if obs_int[i][4] > 0.5:
                            rel = np.array([obs_int[i][5], obs_int[i][6]], dtype=np.float32)
                            rel_norm = float(np.linalg.norm(rel))
                            if action_speed > 1e-8 and rel_norm > 1e-8:
                                visible_alignments.append(
                                    float(
                                        np.dot(
                                            np.array([vx, vy], dtype=np.float32),
                                            rel,
                                        ) / (action_speed * rel_norm)
                                    )
                                )
                    repeat_left = max(1, action_repeat)

                assert cached_actions is not None
                _, _, terminations, truncations, infos = env.step(cached_actions)
                repeat_left -= 1
                ep_steps += 1
                total_steps += 1

                if env._team_sees_prey():
                    ep_visible_steps += 1

                if infos["tactical_state"] == PreyTacticalState.THREATENED:
                    threatened_steps += 1

                capture = infos["capture"]
                ep_max_hold = max(ep_max_hold, capture.hold_counter)
                ep_max_combined = max(
                    ep_max_combined, capture.wall_count + capture.in_range_count
                )

                if any(terminations.values()) or any(truncations.values()):
                    if any(terminations.values()):
                        captures += 1
                    break

            max_holds.append(ep_max_hold)
            max_combined.append(ep_max_combined)
            visible_fracs.append(ep_visible_steps / max(1, ep_steps))
    finally:
        agent.training = prev_training

    return {
        "capture_rate": captures / max(1, episodes),
        "max_hold_mean": float(np.mean(max_holds)) if max_holds else 0.0,
        "max_combined_mean": float(np.mean(max_combined)) if max_combined else 0.0,
        "visible_frac_mean": float(np.mean(visible_fracs)) if visible_fracs else 0.0,
        "visible_alignment_mean": (
            float(np.mean(visible_alignments)) if visible_alignments else 0.0
        ),
        "action_speed_mean": float(np.mean(action_speeds)) if action_speeds else 0.0,
        "threatened_frac": threatened_steps / max(1, total_steps),
    }


def _agent_action_space(agent):
    action_spaces = getattr(agent, "action_spaces", None)
    if action_spaces:
        return action_spaces[0]

    action_space = getattr(agent, "action_space", None)
    if isinstance(action_space, dict) and action_space:
        return next(iter(action_space.values()))

    raise ValueError("Could not determine agent action space")


def _eval_metrics_better(
    current: dict[str, float],
    best: dict[str, float] | None,
) -> bool:
    if best is None:
        return True

    for key in ("capture_rate", "max_hold_mean", "max_combined_mean"):
        if current[key] > best[key]:
            return True
        if current[key] < best[key]:
            return False

    if current["visible_alignment_mean"] > best["visible_alignment_mean"]:
        return True
    if current["visible_alignment_mean"] < best["visible_alignment_mean"]:
        return False

    return current["action_speed_mean"] > best["action_speed_mean"]


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Device: {device}")

    # ── curriculum setup ──────────────────────────────────────────────────
    if args.no_curriculum:
        stages = [CURRICULUM_STAGES[-1]]
    else:
        stages = list(CURRICULUM_STAGES)

    current_stage = 0
    stage_env_kwargs = dict(stages[current_stage]["env_kwargs"])

    # ── environment ───────────────────────────────────────────────────────
    env = build_vec_env(args.num_envs, args.action_repeat, stage_env_kwargs)

    agent_ids = list(env.agents)
    observation_spaces = [env.single_observation_space(agent) for agent in agent_ids]
    action_spaces = [env.single_action_space(agent) for agent in agent_ids]
    action_space_by_agent = dict(zip(agent_ids, action_spaces))

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
        "BATCH_SIZE": 128,
        "O_U_NOISE": True,
        "EXPL_NOISE": 0.25,
        "MEAN_NOISE": 0.0,
        "THETA": 0.15,
        "DT": 1.0,
        "LR_ACTOR": 3e-4,
        "LR_CRITIC": 3e-4,
        "GAMMA": 0.99,
        "MEMORY_SIZE": 500_000,
        "LEARN_STEP": 50,
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
        tournament_size=min(2, INIT_HP["POPULATION_SIZE"]),
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
    best_eval_metrics: dict[str, float] | None = None
    best_eval_path = os.path.join(args.save_dir, "MATD3_best.pt")

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
                agent_total_steps = agent.steps[-1] + steps
                if agent_total_steps < args.random_action_steps:
                    cont_actions = sample_vectorized_random_actions(
                        agent.agent_ids,
                        action_space_by_agent,
                        args.num_envs,
                    )
                else:
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
                if agent.steps[-1] + steps >= args.learning_starts:
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
        eval_agent = pop[int(np.argmax(fitnesses))]
        eval_metrics = evaluate_policy(
            eval_agent,
            args.action_repeat,
            stage_env_kwargs,
            episodes=args.eval_episodes,
            base_seed=args.seed + 10_000,
        )

        if _eval_metrics_better(eval_metrics, best_eval_metrics):
            best_eval_metrics = dict(eval_metrics)
            eval_agent.save_checkpoint(best_eval_path)
            print(
                "  [best-eval] "
                f"cap {eval_metrics['capture_rate']:.0%} | "
                f"hold {eval_metrics['max_hold_mean']:.1f} | "
                f"comb {eval_metrics['max_combined_mean']:.1f} -> "
                f"{best_eval_path}"
            )
        stage_advance_rate = stages[current_stage].get(
            "advance_rate", CURRICULUM_ADVANCE_RATE
        )

        if (
            current_stage < len(stages) - 1
            and len(recent_outcomes) >= CURRICULUM_WINDOW
            and recent_capture_rate >= stage_advance_rate
        ):
            current_stage += 1
            stage_env_kwargs = dict(stages[current_stage]["env_kwargs"])
            print(f"\n*** CURRICULUM ADVANCE -> {stages[current_stage]['label']} "
                  f"(capture rate {recent_capture_rate:.0%}) ***")
            env.close()
            env = build_vec_env(args.num_envs, args.action_repeat, stage_env_kwargs)
            recent_outcomes.clear()

        # ── TensorBoard logging ──────────────────────────────────────────
        writer.add_scalar("train/mean_score", np.mean(mean_scores), total_steps)
        writer.add_scalar("train/elite_fitness", max(fitnesses), total_steps)
        writer.add_scalar("train/episodes", episodes_completed, total_steps)
        writer.add_scalar("train/capture_rate_recent", recent_capture_rate, total_steps)
        writer.add_scalar("train/captures_total", captures, total_steps)
        writer.add_scalar("train/timeouts_total", timeouts, total_steps)
        writer.add_scalar("train/curriculum_stage", current_stage, total_steps)
        writer.add_scalar(
            "train/prey_speed_factor",
            float(stage_env_kwargs["prey_speed_factor"]),
            total_steps,
        )
        writer.add_scalar(
            "train/obstacles_enabled",
            float(bool(stage_env_kwargs["enable_obstacles"])),
            total_steps,
        )
        writer.add_scalar(
            "train/always_visible",
            float(bool(stage_env_kwargs["always_visible"])),
            total_steps,
        )
        writer.add_scalar(
            "train/capture_hold_seconds",
            float(stage_env_kwargs["capture_hold_seconds"]),
            total_steps,
        )
        writer.add_scalar(
            "train/wall_time_min", (time.time() - t_start) / 60, total_steps
        )
        writer.add_scalar("train/replay_buffer_size", len(memory), total_steps)
        writer.add_scalar("eval_fixed/capture_rate", eval_metrics["capture_rate"], total_steps)
        writer.add_scalar("eval_fixed/max_hold_mean", eval_metrics["max_hold_mean"], total_steps)
        writer.add_scalar(
            "eval_fixed/max_combined_mean",
            eval_metrics["max_combined_mean"],
            total_steps,
        )
        writer.add_scalar(
            "eval_fixed/visible_frac_mean",
            eval_metrics["visible_frac_mean"],
            total_steps,
        )
        writer.add_scalar(
            "eval_fixed/visible_alignment_mean",
            eval_metrics["visible_alignment_mean"],
            total_steps,
        )
        writer.add_scalar(
            "eval_fixed/action_norm_mean",
            eval_metrics["action_speed_mean"],
            total_steps,
        )
        writer.add_scalar(
            "eval_fixed/action_speed_mean",
            eval_metrics["action_speed_mean"],
            total_steps,
        )
        writer.add_scalar(
            "eval_fixed/threatened_frac",
            eval_metrics["threatened_frac"],
            total_steps,
        )

        elapsed = time.time() - t_start
        print(
            f"\n--- Steps {total_steps:,} | "
            f"Ep {episodes_completed} | "
            f"Cap {captures} | TO {timeouts} | "
            f"Rate {recent_capture_rate:.0%} | "
            f"{stages[current_stage]['label']} | "
            f"EvalCap {eval_metrics['capture_rate']:.0%} | "
            f"EvalHold {eval_metrics['max_hold_mean']:.1f} | "
            f"EvalComb {eval_metrics['max_combined_mean']:.1f} | "
            f"EvalAlign {eval_metrics['visible_alignment_mean']:.2f} | "
            f"EvalSpeed {eval_metrics['action_speed_mean']:.1f} | "
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
