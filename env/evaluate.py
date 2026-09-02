"""Evaluate a learned MATD3 or behavior-cloned policy on held-out layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from agilerl.algorithms import MATD3

from swarm_ml import build_matd3, load_bc_actors
from train import evaluate_capture_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate swarm capture rate")
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--checkpoint", help="MATD3 .pt checkpoint")
    policy.add_argument("--bc-checkpoint", help="Behavior-cloned actor .pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2_000_000)
    parser.add_argument("--prey-speed-factor", type=float, default=1.0)
    parser.add_argument("--capture-hold-seconds", type=float, default=2.0)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--no-obstacles", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path for JSON metrics")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    if args.checkpoint:
        agent = MATD3.load(args.checkpoint, device=device)
        policy_path = args.checkpoint
        policy_type = "matd3"
    else:
        agent = build_matd3(device=device)
        load_bc_actors(agent, args.bc_checkpoint)
        policy_path = args.bc_checkpoint
        policy_type = "behavior_cloning"

    stage = {
        "prey_speed_factor": args.prey_speed_factor,
        "obstacles_enabled": not args.no_obstacles,
        "capture_hold_seconds": args.capture_hold_seconds,
    }
    capture_rate, mean_steps = evaluate_capture_rate(
        agent,
        stage,
        args.episodes,
        seed_start=args.seed,
        action_repeat=args.action_repeat,
    )
    metrics = {
        "policy_type": policy_type,
        "policy_path": str(policy_path),
        "episodes": args.episodes,
        "seed_start": args.seed,
        "prey_speed_factor": args.prey_speed_factor,
        "obstacles_enabled": not args.no_obstacles,
        "capture_hold_seconds": args.capture_hold_seconds,
        "action_repeat": args.action_repeat,
        "capture_rate": capture_rate,
        "mean_core_steps": mean_steps,
    }
    print(json.dumps(metrics, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
