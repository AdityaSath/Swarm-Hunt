"""Behavior-clone the scripted expert into the shared MATD3 actor."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from swarm_env.environment import OBS_SIZE
from swarm_ml import build_matd3, save_bc_actor


class ExpertDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with h5py.File(self.path, "r") as file:
            if file.attrs.get("format") != "swarm_hunt_expert_v1":
                raise ValueError(f"Unsupported expert dataset: {path}")
            if int(file.attrs["obs_size"]) != OBS_SIZE:
                raise ValueError(
                    f"Dataset observation size {file.attrs['obs_size']} != {OBS_SIZE}"
                )
            # A 500-episode dataset is typically a few hundred MB. Loading it
            # once is substantially faster than random single-row HDF5 reads
            # on every epoch and keeps DataLoader behavior deterministic.
            self.observations = torch.from_numpy(
                np.asarray(file["observations"], dtype=np.float32)
            )
            self.actions = torch.from_numpy(
                np.asarray(file["actions"], dtype=np.float32)
            )
            self.episode_indices = np.asarray(
                file["episode_indices"], dtype=np.int32
            )

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.observations[index], self.actions[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior-clone expert controller")
    parser.add_argument("--dataset", default="data/expert_capture.h5")
    parser.add_argument("--output", default="models/BC/formation_actor.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not 0.0 < args.validation_split < 1.0:
        raise ValueError("--validation-split must be between 0 and 1")
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    dataset = ExpertDataset(args.dataset)
    unique_episodes = np.unique(dataset.episode_indices)
    if len(unique_episodes) < 2:
        raise ValueError("Expert dataset must contain at least two episodes")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_episodes)
    val_episode_count = max(1, int(len(unique_episodes) * args.validation_split))
    val_episodes = unique_episodes[:val_episode_count]
    validation_mask = np.isin(dataset.episode_indices, val_episodes)
    train_indices = np.flatnonzero(~validation_mask).tolist()
    val_indices = np.flatnonzero(validation_mask).tolist()
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    agent = build_matd3(device=device)
    actor = next(iter(agent.actors.values()))
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        actor.train()
        train_loss = 0.0
        train_samples = 0
        for observations, actions in train_loader:
            observations = observations.to(device)
            actions = actions.to(device)
            predictions = actor(observations)
            loss = loss_fn(predictions, actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(observations)
            train_samples += len(observations)

        actor.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for observations, actions in val_loader:
                observations = observations.to(device)
                actions = actions.to(device)
                loss = loss_fn(actor(observations), actions)
                val_loss += float(loss.item()) * len(observations)
                val_samples += len(observations)

        mean_train = train_loss / max(1, train_samples)
        mean_val = val_loss / max(1, val_samples)
        print(
            f"epoch={epoch:03d}/{args.epochs}  "
            f"train_mse={mean_train:.6f}  val_mse={mean_val:.6f}"
        )
        if mean_val < best_val:
            best_val = mean_val
            save_bc_actor(
                agent,
                args.output,
                {
                    "dataset": str(Path(args.dataset)),
                    "epoch": epoch,
                    "validation_mse": mean_val,
                    "seed": args.seed,
                },
            )

    print(f"Saved best behavior-cloned actor to {args.output} (MSE={best_val:.6f})")


if __name__ == "__main__":
    main()
