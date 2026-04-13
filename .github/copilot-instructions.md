# Copilot instructions for Swarm-Hunt

This repo is a small Pygame-based swarm prototype. The active implementation lives under `env/`.

## What to edit
- Primary working area: `env/swarm_env/`
- Entry point: `env/main.py`
- Runtime docs: `env/README.md`
- Dependencies: `env/requirements.txt`

## Architecture overview
- `env/swarm_env/environment.py` is the main environment controller.
- `env/swarm_env/drone.py` defines passive agent state, update/draw, and heading/thrust/steer motion.
- `env/swarm_env/arena.py` handles arena bounds and wall bounce/clamping.
- `env/swarm_env/obstacle.py` defines static rectangular obstacles.
- `env/swarm_env/spatial.py` defines the `NeighborFinder` protocol and the current `DistanceBasedNeighborFinder`.
- `env/swarm_env/config.py` centralizes tunable constants like `DRONE_SPEED`, `DRONE_PERCEPTION_RANGE`, and `OBSTACLE_POSITIONS`.

## Key patterns and conventions
- The environment is designed as an RL-ready wrapper, not a full RL agent implementation.
- `Environment.step(actions)` returns `(observations, rewards, terminations, truncations, infos)`.
- `actions` is a dict mapping `agent_id -> (thrust, steer)`.
  - `thrust` is effectively forward-only in code: values are clamped to `[0.0, 1.0]`.
  - `steer` is clamped to `[-1.0, 1.0]`.
- `actions=None` triggers demo wandering behavior with randomly generated forward thrust and steer values.
- Missing agent IDs in the action dict leave the drone's current `thrust`/`steer` unchanged.
- Observations are local-only and computed by `_compute_observations()` every step.
  - Each observation contains `obstacles`, `neighbors`, `boundaries`, and `self_state`.
- `NeighborFinder` is intentionally pluggable; new spatial indexes should implement the same signatures.
- Drones are state containers only; all collision, physics, and observation logic lives in `Environment`.

## Important implementation details
- `Environment.reset(seed)` recreates obstacles and drones and returns a dict of per-agent observations.
- `Environment._physics_step()` handles movement, obstacle repulsion, arena clamping, obstacle collision, and drone-drone separation.
- `Drone.update(dt)` applies `steer` to rotate heading and `thrust` to set velocity along that heading.
- `Environment.get_observation(agent_id)` recomputes observations on demand and is not a cached accessor.
- `env/main.py` runs a demo loop with Pygame and uses `env.step()` with no explicit actions.

## Runtime commands
- Install dependencies: `pip install -r env/requirements.txt`
- Run the demo: `python env/main.py`

## What not to assume
- There is no existing integrated test suite in `env/`; use the demo loop or add targeted tests.
- The repo does not currently expose a stable gym/PettingZoo environment beyond the local API in `Environment`.
- `MARLtest/` contains separate verification scripts and should not be treated as the main runtime for `env/swarm_env/`.

## Good first edits
- Keep the local-observation design intact when extending sensors.
- Preserve the `NeighborFinder` abstraction if modifying neighbor lookup.
- Keep physics/collision helper functions in `environment.py` grouped and clear.
- Use `pygame.math.Vector2` consistently for positions and velocities.
