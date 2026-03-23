# Drone & Environment Reference for RL

## 1. Environment API

| Method | Signature | Returns |
|--------|------------|---------|
| `reset(seed=None)` | `env.reset(seed)` | `(observations, infos)` |
| `step(actions)` | `env.step(actions)` | `(observations, rewards, terminations, truncations, infos)` |
| `get_observation(agent_id)` | `env.get_observation(i)` | Observation dict for agent `i` |

---

## 2. Action Space

**Format:** `actions: dict[int, tuple[float, float]]` — maps `agent_id` → `(thrust, steer)`.

| Parameter | Range | Description |
|-----------|--------|--------------|
| **thrust** | `[0.0, 1.0]` | Forward speed (0 = stop, 1 = max). **No backward.** |
| **steer** | `[-1.0, 1.0]` | Turn rate (-1 = left, 0 = straight, 1 = right) |

**Clamping:** Values outside these ranges are clamped.

**Example:**
```python
actions = {
    0: (1.0, 0.0),   # agent 0: full forward, no turn
    1: (0.5, -0.5),  # agent 1: half speed, turn left
}
obs, rewards, terms, truncs, infos = env.step(actions)
```

---

## 3. Observation Space (per agent)

Each observation is a dict with **local-only** info (within `perception_range`):

```python
{
    "obstacles": [
        {
            "relative_pos": (dx, dy),   # offset from this drone
            "distance": float,
            "rect": pygame.Rect,        # obstacle bounds
        },
        ...
    ],
    "neighbors": [
        {
            "relative_pos": (dx, dy),
            "distance": float,
            "velocity": (vx, vy),
            "id": int,                 # neighbor's agent_id
        },
        ...
    ],
    "boundaries": {
        "left": float,    # distance to left wall
        "right": float,
        "top": float,
        "bottom": float,
    },
    "self_state": {
        "position": (x, y),
        "velocity": (vx, vy),
        "heading": float,   # radians (0 = right, π/2 = down)
    },
}
```

---

## 4. Drone State (Internal)

| Attribute | Type | Description |
|-----------|------|-------------|
| `position` | `Vector2(x, y)` | World coordinates |
| `velocity` | `Vector2(vx, vy)` | Pixels per second |
| `heading` | `float` | Facing angle in **radians** (0 = right, π/2 = down) |
| `thrust` | `float` | Current thrust input [0, 1] |
| `steer` | `float` | Current steer input [-1, 1] |
| `collision_radius` | `float` | 15 px |
| `perception_range` | `float` | 120 px |

---

## 5. Physics Model

**Per timestep (`dt = 1/60` s):**

1. **Heading:** `heading += steer * DRONE_MAX_TURN_RATE * dt`
2. **Velocity:** `velocity = thrust * DRONE_SPEED * (cos(heading), sin(heading))`
3. **Obstacle repulsion:** Repulsion force (1/distance²) is added to velocity; speed clamped to DRONE_SPEED
4. **Position:** `position += velocity * dt`
5. **Collisions:** On overlap (walls, obstacles, other drones), position is corrected and velocity set to 0 (no bounce)

**Constants:**
- `DRONE_SPEED` = 80 px/s
- `DRONE_MAX_TURN_RATE` = π/3 rad/s (~60°/s)

---

## 6. Coordinate System

- **Origin:** Top-left of arena
- **X:** Right = positive
- **Y:** Down = positive (Pygame convention)
- **Heading:** 0 rad = right, π/2 rad = down

---

## 7. Config Constants

| Constant | Value |
|----------|-------|
| Arena size | 1200 × 900 px |
| Drone count | 10 |
| Drone radius | 15 px |
| Drone max speed | 80 px/s |
| Max turn rate | π/3 rad/s |
| Perception range | 120 px |
| Obstacle repulsion range | 100 px |
| FPS / dt | 60 / 0.0167 s |

---

## 8. Notes for RL Team

- **Partial actions:** Omitted agents keep their previous thrust/steer
- **Rewards:** Currently all 0 (define your own)
- **Terminations/truncations:** Currently all False
- **`infos["agents"]`:** List of active agent IDs
- **No backward motion:** Thrust is clamped to [0, 1] only
