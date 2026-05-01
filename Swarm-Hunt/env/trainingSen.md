# Training Scenarios and Variables — Brainstorm

This document catalogs candidate training scenarios, environment/agent variables to randomize, curriculum ideas, and evaluation metrics for training predator agents (centralized or multi-agent) to pursue, flank, and capture a prey.

## Goals
- Train robust predator policies that (a) reliably capture prey, (b) coordinate to flank/surround, and (c) generalize across arena layouts and sensory noise.
- Explore how starting conditions and environment variability affect emergent tactics.

## Key variables to change (randomize)
- Starting positions
  - Predator spawn: clustered, evenly spaced around arena edge, random uniform, or biased to one quadrant.
  - Prey spawn: center, edge, random, or adversarial (near cover/obstacle).
- Agent counts
  - Number of predators (N): small (2–3), medium (4–6), and larger (>6).
  - Number of prey: single vs multiple prey (cooperative prey or independent agents).
- Arena / environment
  - Size: small / medium / large arenas (scale DRONE_SPEED or time budget accordingly).
  - Topology: open, walls/rooms, narrow corridors, islands, cluttered with obstacles.
  - Dynamic obstacles: moving obstacles vs static obstacles.
- Physics / dynamics
  - Agent max speed (DRONE_SPEED) and prey speed (PREY_SPEED) ratios.
  - Turn rate / inertia / control latency.
- Sensing & observability
  - Partial observability: limited-radius sensors, occlusion by obstacles.
  - Sensor noise: Gaussian noise on positions/velocities, dropped observations.
  - Delay: action or observation delay (simulated latency).
- Communication & team info 
  - Full state sharing (centralized), local observations only (decentralized), or limited-bandwidth comms (messages per timestep).
- Reward / task specification
  - Single-team reward (sum of team reward), per-agent reward, or shaped rewards for flanking/encirclement.
  - Sparse vs dense rewards (sparse: capture only; dense: distance-to-prey penalties, gap-seeking bonuses).

## Scenario templates

- Scenario A — Open Arena Random Start
  - Arena: open, medium size.
  - Start: predators and prey random uniform.
  - Goal: maximize capture rate within time limit.
  - Purpose: baseline pursuit and cooperative coordination.

- Scenario B — Edge Ambush
  - Predators start clustered on one side/edge, prey spawns opposite edge or center.
  - Purpose: encourage spreading and flanking.

- Scenario C — Corridor Chase
  - Arena with narrow corridor(s) connecting rooms.
  - Predators must intercept prey in chokepoints; obstacles occlude vision.
  - Purpose: teach coordinated interception and blocking.

- Scenario D — Cluttered Environment
  - Many static obstacles and blocking regions; random obstacle placements each episode.
  - Purpose: spatial reasoning, navigation, and trapping behaviors.

- Scenario E — Moving Obstacles / Dynamic Environment
  - Include moving objects (e.g., neutral vehicles) that may block paths.
  - Purpose: robustness to non-stationary world.

- Scenario F — Multi-Prey / Decoys
  - Two prey: one cooperative (evasive), one random decoy; predators must prioritize.
  - Purpose: test target selection and allocation.

- Scenario G — Limited Sensing / Occlusion
  - Predators only see within radius; obstacles block LOS.
  - Purpose: encourage communication or predictive tracking.

## Curriculum / Training schedule ideas
- Stage 0: Simple open-area pursuit (dense reward: negative distance to prey) to bootstrap basic pursuit.
- Stage 1: Add reward shaping for velocity alignment and mild penalty for collisions.
- Stage 2: Introduce obstacles and partial observability with domain randomization (arena size, prey speed).
- Stage 3: Increase prey agility / adversarial prey policy; transition to sparse-reward capture objective.
- Stage 4: Transfer to full randomized evaluation (test-time generalization across unseen maps).

Automated curriculum tuning
- Automatically increase difficulty when success rate > threshold (e.g., 80%) for several consecutive evaluation windows.

## Reward engineering suggestions
- Sparse-team reward: +1 when prey captured, 0 otherwise — encourages final outcome but training may be slow.
- Dense shaping terms (use with care):
  - negative distance-to-nearest-predator for prey (or negative average distance for predator team) scaled to episode length.
  - small reward for reducing prey speed or constraining prey movement.
  - bonus for surrounding: reward when prey is within polygon/hull formed by predators or when multiple predators are within capture radius.
- Penalties: collisions with obstacles or other agents, leaving arena bounds, or excessive accelerations.

## Evaluation metrics
- Capture rate: fraction of episodes resulting in capture within time budget.
- Time-to-capture: mean/median steps to capture (for successful episodes).
- Success under perturbations: capture rate when varying prey speed, starting positions, and with sensor noise.
- Coordination metrics:
  - Coverage of arena boundary or encirclement score (fraction of angular sectors around prey occupied by predators).
  - Minimum pairwise distance among predators (to detect excessive bunching).
- Sample efficiency: timesteps to reach X% success.

## Experimental matrix (example grid)

| Variable | Values to try |
|---|---|
| Predator count | 2, 4, 6 |
| Prey speed (ratio) | 0.8, 1.0, 1.2 |
| Arena size | small, medium, large |
| Obstacles | none, sparse, dense |
| Observability | full, limited radius, occluded |
| Reward type | dense-shaping, sparse-end |

Run a factorial sweep for a small subset first (e.g., predator count × prey speed × obstacles) before expanding the grid.

## Implementation notes / machine setup
- For reproducibility: seed RNGs for env, numpy, and torch; log seeds with each experiment.
- Use vectorized environments (DummyVecEnv or SubprocVecEnv) for faster data collection; for macOS prefer DummyVecEnv if SDL/graphics issues arise.
- Headless training: when training on a server, disable rendering and use headless-only environment variants (no pygame init in worker processes) or set SDL environment variables.
- Domain randomization: randomize both visual and non-visual parameters at training time to improve generalization.

## Centralized vs Decentralized training
- Centralized (single policy outputs all agents): simpler to implement, often sample-efficient, useful as a strong baseline. May not scale to large N.
- Decentralized (one policy per agent or shared policy with observation masking): better scaling, enforces local decision-making and robustness to agent failures.
- Hybrid: centralized critic with decentralized actors (CTDE) commonly used in MARL.

## Baselines and sanity checks
- Scripted policies: pursuit-only and simple flanking heuristics (already present) as behavior baselines.
- Random policy: sanity check for reward and capture rates.
- Centralized PPO baseline (existing) as a reference point; compare with shared-policy PPO and/or MADDPG.

## Logging & checkpoints
- Save models periodically and evaluate on held-out deterministic scenarios.
- Record videos (or frame dumps) for qualitative inspection of learned behaviors.

## Quick experiments to run first
1. Baseline pursuit: open arena, 4 predators, prey speed 0.8×, dense reward for distance — train until capture rate > 80%.
2. Add obstacles: sparse static obstacles randomized each episode and re-run same hyperparameters; compare capture rate drop.
3. Partial observability: limit predator sensing radius and see whether team learns to coordinate or needs communication.

## Notes / next steps
- Convert top scenarios into deterministic unit tests (non-rendering) that assert minimum success rates after short fine-tuning runs. This helps guard regressions.
- Consider adding a small evaluation harness that runs batches of deterministic scenarios and summarizes metrics to CSV for plotting.

---

Place this file under `env/` and iterate: pick 3–5 core scenarios to implement as environment presets and create a small experiment runner that automates sweeps.
