# 2D Swarm Environment

A modular Pygame-based 2D swarm prototype environment with a bounded arena, obstacles, and multiple hexagon-shaped drones. Designed for extensibility (flocking, pathfinding, RL) rather than a one-off visual demo.

This README summarizes the environment, recent changes made during development, what currently works, and recommended next steps (behaviour cloning and curricula) to make RL capture reliable.

## Setup

From the repository root, in the project's Python virtual environment:

```bash
pip install -r requirements.txt
```

Run the visual demo UI:

```bash
python main.py
```

- Space: Pause / resume
- Escape: Quit

## Current status (verified)

- Unit tests: all environment tests pass locally (6 passed).
- Scripted baseline: evaluated with `env/scripts/eval_scripted.py` and a deterministic seed sweep. The locked weights captured 48/50 seeded episodes (mean capture steps ≈ 575) and short unseeded runs reliably capture most episodes.
- Locking/eval tools: added `env/locked_policy.py`, `env/scripts/lock_policy.py` and `env/scripts/eval_locked.py` to save and evaluate "locked" policies (either the scripted policy or copied SB3 archives).

## Locked agent settings

The current baseline is intentionally conservative: agents close distance first, then blend into flanking once they are near the prey. These constants live in `env/swarm_env/config.py`.

- `MIN_PREDATOR_CONTRIBUTORS = 4`: restores the intended capture rule where 4 contributors can capture and 3 cannot.
- `PURSUIT_WEIGHT = 0.82`, `FLANK_WEIGHT = 0.18`, `INERTIA_WEIGHT = 0.08`: enough pursuit pressure to avoid orbiting too early while still retaining a flank target.
- `FLANK_RADIUS_MULT = 0.9`: keeps flank targets just inside the capture contribution radius.
- `PHI_ESCAPE_MAX = 90°`, `T_HOLD = 2`: keeps capture achievable for training/debugging while requiring a stable surround.

## Grid search behavior

The hard-coded demo policy uses a 3x3 arena grid before capture. Predators pick random search targets across the grid, get rewarded for first visits and broad coverage, and keep moving with separation pressure so they do not all stack in one cell. When any predator shares the prey's grid cell, that cell becomes known and the group converges there, then blends pursuit with flanking around the prey. The prey remains rule-based and avoidant.

For RL, use the hybrid action wrapper instead of raw velocity control. The low-level controller is still hard-coded, but the PPO policy chooses high-level intent per predator:
- `target_x`, `target_y`: where to search before the prey grid is discovered.
- `flank_angle`: which approach angle to use after discovery.

Train hybrid intent:

```bash
PYTHONPATH="$(pwd)/env" .venv/bin/python env/train_sb3_central.py --hybrid-actions --timesteps 200000 --n-envs 4 --logdir ./sb3_logs_hybrid
```

View a hybrid-trained model:

```bash
PYTHONPATH="$(pwd)/env" .venv/bin/python env/view_policy.py --model ./sb3_logs_hybrid/ppo_central.zip --hybrid-actions --deterministic
```

Additional shaping rewards support this behavior:
- New grid-cell discovery and team grid coverage.
- Same-cell prey discovery and convergence toward the discovered prey grid.
- Flanking diversity for contributors approaching the prey from different angular sectors.

Artifacts saved during experiments (examples):
- `scripted_demos_50.npz` — 50 scripted episodes (actions only).
- SB3 runs:
	- `sb3_logs_retrain_smoke/ppo_central.zip` (50k smoke)
	- `sb3_logs_retrain_long/ppo_central.zip` (200k)
	- `sb3_logs_retrain_warm/ppo_central.zip` (200k with scripted warm-start)

Note: the trained SB3 models from these runs did not reliably capture in short deterministic evals (they often truncated); the environment and scripted baseline are functioning correctly.

## What changed (high level)

- Demo/policy:
	- `scripted_actions()` and demo mode now use the same pursuit + flank policy, so demos and behavior-cloning data come from the same baseline.
	- Scripted demo behavior now searches the 3x3 grid randomly, then flocks to the prey's discovered grid cell.
	- Demo visualization now shows flank targets and assignment lines.

- Capture & rewards:
	- Added blocked-move capture test (`_is_prey_blocked`) to complement angular-gap capture.
	- Added per-agent shaping: pursuit shaping, dispersion-phase rewards, see-prey bonuses, AND flank/contributor bonuses.
	- Added grid discovery, grid coverage, convergence, and flank-diversity rewards.
	- Fixed the see-prey team bonus so it is actually applied to agent rewards.

- RL tooling:
	- Centralized Gym wrapper: `env/gym_centralized.py`.
	- SB3 starter and warm-start support: `env/train_sb3_central.py` (now supports scripted-action warm start).
	- Evaluation tools: `env/eval_headless.py`, `env/evaluate_policy.py`, scripted evaluators and locked-policy CLI.

## How to lock & evaluate a policy

Lock the scripted policy:

```bash
PYTHONPATH="$(pwd)/env" .venv/bin/python env/scripts/lock_policy.py --name scripted_v1 --source scripted
```

Lock an SB3 archive (copies the .zip into `models/` and writes a manifest):

```bash
PYTHONPATH="$(pwd)/env" .venv/bin/python env/scripts/lock_policy.py --name ppo_warm_v1 --source sb3_logs_retrain_warm/ppo_central.zip
```

Evaluate a locked manifest:

```bash
PYTHONPATH="$(pwd)/env" .venv/bin/python env/scripts/eval_locked.py --manifest $(pwd)/models/scripted_v1.json --episodes 20
```

## Recommended next steps (to make RL policies capture reliably)

1) Behavior cloning warm-start (recommended)
	 - Collect centralized observation → action pairs from the scripted policy (augment `collect_scripted_demos.py` to save observations as well).
	 - Train a supervised policy (BC) on these pairs until the actor reproduces scripted actions well.
	 - Fine-tune the BC-initialized policy with PPO for 100k–200k steps.

2) Curriculum / reward-scaling
	 - Start with an easier environment (reduce `PREY_SPEED`, increase `CAP_BLOCK_DIST`, or increase `FLANK_REWARD`) and gradually anneal to the final difficulty.
	 - Alternatively, shape rewards stronger initially (increase `SEE_PREY_REWARD`, `PURSUIT_REWARD_COEF`) then decay.

3) More / diverse demonstrations
	 - Collect additional scripted demos across varied initial positions and obstacle layouts to improve BC coverage.

4) Hyperparameter and architecture sweeps
	 - Try more expressive policies (larger MLP), higher entropy / larger clip ranges, or more parallel envs to improve sample efficiency.

If you want, I can implement the BC pipeline (data collection → BC trainer → PPO fine-tune) and run it end-to-end. That is the fastest way to turn the scripted capture behavior into a learned policy.

## Files & commands reference

- Important files:
	- `env/swarm_env/environment.py` — main sim and `scripted_actions()`
	- `env/swarm_env/config.py` — constants to tune
	- `env/gym_centralized.py`, `env/train_sb3_central.py` — RL wrapper + trainer
	- `env/scripts/collect_scripted_demos.py`, `env/scripts/eval_scripted.py` — demos & scripted eval
	- `env/scripts/lock_policy.py`, `env/scripts/eval_locked.py` — lock & eval helpers

- Quick reproduce commands (from repo root):

```bash
# collect scripted demos (actions only)
PYTHONPATH="$(pwd)" .venv/bin/python env/scripts/collect_scripted_demos.py --episodes 50 --max-steps 2000 --out scripted_demos_50.npz

# short smoke training (centralized PPO)
PYTHONPATH="$(pwd)" .venv/bin/python env/train_sb3_central.py --timesteps 50000 --n-envs 1 --logdir ./sb3_logs_retrain_smoke

# evaluate a saved model (headless)
PYTHONPATH="$(pwd)" .venv/bin/python env/eval_headless.py --model ./sb3_logs_retrain_long/ppo_central.zip --episodes 20 --max-steps 2000 --deterministic

# lock & evaluate a policy
PYTHONPATH="$(pwd)/env" .venv/bin/python env/scripts/lock_policy.py --name ppo_warm_v1 --source sb3_logs_retrain_warm/ppo_central.zip
PYTHONPATH="$(pwd)/env" .venv/bin/python env/scripts/eval_locked.py --manifest $(pwd)/models/ppo_warm_v1.json --episodes 10
```

---

If you'd like, I can now:
- Implement the BC data collector (save obs+actions) and a minimal BC trainer and run a small end-to-end experiment, or
- Add a short section to this README explaining how to tune a few specific constants for faster learning.

Tell me which and I'll proceed.
