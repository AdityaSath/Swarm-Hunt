# 2D Swarm Environment

A modular Pygame-based 2D swarm prototype environment with a bounded arena, obstacles, and multiple hexagon-shaped drones. Designed for extensibility (flocking, pathfinding, RL) rather than a one-off visual demo.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

- **Space**: Pause / resume
- **Escape**: Quit

---

## What's Been Built

### 1. Base Environment (Initial Plan)

- **Arena** — 800×600 bounded rectangle with `contains()`, `clamp()`, `get_bounds()`
- **Obstacles** — 10 static rectangles (6 small 30×30, 4 large 80×60)
- **Drones** — Hexagon agents with position, velocity, heading, collision radius
- **Physics** — Bounce off walls, obstacles, and each other (no stopping)
- **Vision cone** — 45° half-angle, configurable, drawn for each drone

### 2. Drone as Independent Agent

- **State only** — Drones hold: `position`, `velocity`, `heading`, `collision_radius`, `perception_range`
- **No world access** — Drones do not read full world state
- **Limited sensing** — `perception_range` (120 px) caps what they can "see"

### 3. Local Neighbor Lookup

- **`swarm_env/spatial.py`** — Pluggable neighbor search
- **`NeighborFinder`** — Protocol with `find_drone_neighbors()` and `find_nearby_obstacles()`
- **`DistanceBasedNeighborFinder`** — O(n²) distance-based implementation
- **Swappable** — Add `GridNeighborFinder` or quadtree later; pass into `Environment(neighbor_finder=...)` without changing env logic

### 4. RL-Ready API

- **`reset(seed)`** → `(observations, infos)`
- **`step(actions)`** → `(observations, rewards, terminations, truncations, infos)`
- **`actions`** — `agent_id -> (vx, vy)`; missing agents keep current velocity
- **`get_observation(agent_id)`** — Local observation for one agent

### 5. Separation of Concerns

- **Environment** — Applies actions, runs physics, computes observations
- **Drones** — State containers only; no decision logic
- **Physics** — `_apply_actions()`, `_physics_step()` isolated from observation logic

---

## Behavior

| Mode | Behavior |
|------|----------|
| **Demo** (`step()` no args) | Drones keep current velocity; bounce off walls, obstacles, and each other |
| **With actions** (`step({0: (50,0), ...})`) | Specified drones get new velocities; others unchanged |
| **Policies** | None implemented yet; behavior is unchanged from before the refactor |

---

## Observations: Local Only

Each agent's observation includes only what is within `perception_range`:

- **`obstacles`** — Obstacles in range: `{relative_pos, distance, rect}`
- **`neighbors`** — Other drones in range: `{relative_pos, distance, velocity, id}`
- **`boundaries`** — Distances to arena edges
- **`self_state`** — Own `position`, `velocity`, `heading`

No global world state is exposed to agents.

---

## Swapping in a Grid Later

Yes. The design supports swapping the neighbor finder cleanly:

```python
# Current (distance-based)
env = Environment(neighbor_finder=DistanceBasedNeighborFinder())

# Later (grid-based)
env = Environment(neighbor_finder=GridNeighborFinder(cell_size=50))
```

`Environment` only depends on the `NeighborFinder` interface. Implement `find_drone_neighbors()` and `find_nearby_obstacles()` with the same signatures and return types, and you can swap without touching environment or drone code.

---

## File Layout

```
swarm_env/
├── arena.py        # Bounded arena
├── obstacle.py     # Static obstacles
├── drone.py        # State container (position, velocity, heading, radius, perception_range)
├── spatial.py      # NeighborFinder protocol + DistanceBasedNeighborFinder
├── environment.py  # reset, step(actions), _physics_step, _compute_observations
└── config.py       # Tunables
main.py             # Pygame loop, step() with no actions (demo)
```

---

## API Reference

### RL-Ready API

```python
# Reset
observations, infos = env.reset(seed=42)

# Step: actions = agent_id -> (vx, vy). None = keep current velocity.
observations, rewards, terminations, truncations, infos = env.step(actions)

# Single-agent observation
obs = env.get_observation(agent_id)
```

### Observation Structure

```python
obs = env.get_observation(0)
# obs["obstacles"]   # [{relative_pos, distance, rect}, ...] in range
# obs["neighbors"]   # [{relative_pos, distance, velocity, id}, ...] in range
# obs["boundaries"]  # {left, right, top, bottom} to arena edges
# obs["self_state"]  # {position, velocity, heading}
```

### Global Query API (debugging)

```python
env.get_drone_positions()   # list of (x, y)
env.get_drone_velocities()  # list of (vx, vy)
env.get_obstacles()         # list of pygame.Rect
env.get_neighbors(drone_id, radius=None)  # uses drone.perception_range if radius is None
```

---

## Summary

- **Drones** — State containers; no decision logic.
- **Observations** — Local only; limited by `perception_range`.
- **Neighbor lookup** — Pluggable; grid/quadtree can replace distance-based search.
- **Policies** — Not implemented; behavior is still bouncing with random initial velocities.
