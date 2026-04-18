# V1 Multi-Agent Pursuit Environment

A 2D continuous multi-agent pursuit environment with:
- Learning agents: predator drones
- Scripted agent: one prey
- API target: PettingZoo Parallel (`PursuitParallelEnv`)
- Trainer target: AgileRL

This README reflects the current code in `env/swarm_env/*`.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Demo controls (`main.py`):
- `Space`: pause/resume
- `R`: reset episode
- `Esc`: quit

### Trained MATD3 demo (`demo.py`)

Runs pygame with policies loaded from a checkpoint (default: most recent `models/MATD3/*.pt`).

Controls: same as above (`Space`, `R`, `Esc`).

Important: training uses **`action_repeat=4`** on `PursuitParallelEnv` (see `train.py`). The demo repeats each policy action for the same number of physics steps so behavior matches training and motion is stable. Override only if your checkpoint was trained with a different repeat.

CLI (common options):

| Flag | Default | Notes |
|------|---------|--------|
| `--checkpoint` | latest in `models/MATD3/` | Path to `.pt` |
| `--action-repeat` | `4` | Physics steps per policy decision; match `train.py` |
| `--prey-speed-factor` | `1.0` | Scales prey speed in the env |
| `--episodes` | `0` | Auto-exit after N completed episodes (`0` = run until closed) |

Example:

```bash
python demo.py --checkpoint models/MATD3/MATD3_final_12345.pt --action-repeat 4
```

## Current Implemented Defaults

Defined in `swarm_env/config.py`:
- Predators: `DRONE_COUNT = 8`
- Prey: `1`
- `r_prey = 2 * r_pred`
- `v_prey = 1.5 * v_pred`
- `R_SENSE = 8 * r_prey` (team prey sensing / teammate slots)
- `R_DANGER = 4 * r_prey` (FREE ↔ THREATENED, nearest-predator distance)
- `R_CAP = 2.5 * r_prey` (legacy base; capture ring is scaled from this)
- `R_CAPTURE_RANGE = 1.2 * R_CAP` (predators inside this radius count toward capture)
- `CAPTURE_HOLD_SECONDS = 2.0` → `CAPTURE_HOLD_STEPS = 2 * FPS` (consecutive steps the hold must stay valid)
- `COMBO_CAPTURE_NEED = 4` (need **walls intersecting blue circle + drones inside `R_CAPTURE_RANGE`** ≥ this to build the hold)
- `T_HIDE_MAX = 20`
- `DT = 1 / FPS`, `FPS = 60`
- `MAX_STEPS = 30 * FPS` (30 seconds)

## Architecture

### Core env (training logic)
- File: `swarm_env/environment.py`
- Class: `Environment`
- Uses integer agent indices (`0..N-1`)
- No PettingZoo dependency

Step order:
1. Apply predator desired velocities
2. Scripted prey policy
3. Physics and collisions
4. Distance capture check + tactical FSM
5. Reward computation
6. Observation assembly

### Thin PettingZoo adapter
- File: `swarm_env/parallel_env.py`
- Class: `PursuitParallelEnv`
- Maps `predator_i <-> i`
- Exposes `ParallelEnv` API + Gymnasium spaces
- **`action_repeat`**: each `step()` applies the same action for `action_repeat` physics steps (default in training: `4`; stops early if the episode terminates)

## How Predators Work

Predators are kinematic agents (`swarm_env/drone.py`):
- Action: `(vx_desired, vy_desired)`
- Speed clipping: `||v|| <= DRONE_SPEED`
- Integration: `position += velocity * DT`
- Predators cannot pass through walls, obstacles, or other predators

### Demo-mode movement
When `actions=None`, the env uses smooth random wandering:
- Keeps a sampled velocity for about 1 second
- Then samples a new direction/speed

For RL training, always pass explicit actions.

## How Prey Works (Scripted)

File: `swarm_env/prey.py`

Priority policy:
1. If threatened: flee **away from the nearest predator** (velocity in the outward radial direction)
2. If threatened and an obstacle is nearby: may enter obstacle to hide
3. If hidden: remain hidden up to `T_HIDE_MAX`, then forced exit (flee from nearest predator)
4. Otherwise: gentle wander away from the predator cluster centroid

Physics rule:
- Prey can pass through obstacles
- Prey is clamped by arena borders

## Capture Logic (Distance Ring + Hold)

File: `swarm_env/capture.py`

Capture is **distance-based** (no angular gap math).

- **`walls_intersecting_capture_circle`**: count arena edges (0–4) whose perpendicular distance to the prey is at most **`R_CAPTURE_RANGE`** (the blue disk touches that wall).
- **`predators_in_capture_range`**: count predator centers within **`R_CAPTURE_RANGE`** of the prey.
- If **`walls + drones >= COMBO_CAPTURE_NEED`** (default 4), the **hold counter** increments; otherwise it resets to `0`.
- When the hold counter reaches **`CAPTURE_HOLD_STEPS`** (~2 s at 60 FPS), tactical state becomes **`CAPTURED`**.

**`predators_in_capture_range`** also supplies contributor indices for **`CONTRIBUTOR_BONUS`** (same radius **`R_CAPTURE_RANGE`**).

## Tactical State Machine

Prey tactical states (`PreyTacticalState`):
- `FREE` — nearest predator beyond `R_DANGER` (with hysteresis via `MARGIN_THREATENED` when leaving `THREATENED`)
- `THREATENED` — nearest predator within `R_DANGER`
- `CAPTURED` — terminal; set when the distance hold completes

Global episode state (`EpisodeState`):
- `IN_PURSUIT`
- `CAPTURED`
- `TIMEOUT`

Each `step` info dict includes **`capture`**: a **`CaptureStatus`** named tuple `(in_range_count, contributor_indices, hold_counter, wall_count)`.

## Rewards (Per-Drone Reward)

Each predator now receives its own reward based on its own outcome and
contribution.

Included terms:
- Terminal:
  - all predators on capture: `+5` (`REWARD_CAPTURE_TEAM`)
  - extra capture contributor bonus: `+5` (`REWARD_CAPTURE_CONTRIBUTOR`)
  - timeout: `-2` (`REWARD_TIMEOUT`) for every predator
- Shared team progress:
  - `FREE → THREATENED`: `+0.25` (`REWARD_THREATENED`) for every predator
  - positive change in `(walls + drones in ring)`: `+0.15` per step of improvement (`REWARD_COMBINED_PROGRESS`)
  - hold-counter progress while a valid capture combo is maintained: `+0.02` per hold step (`REWARD_HOLD_PROGRESS`)
- Penalties:
  - obstacle collision: `-0.25` for the predator that hit
  - predator collision: `-0.10` for each predator involved
  - idle penalty: `0.0` per-step when speed below `IDLE_SPEED_THRESHOLD`
- Shaping:
  - each predator's own prey-distance delta, clipped by `DIST_SHAPING_CLIP`
- Optional:
  - tiny per-step **`CONTRIBUTOR_BONUS`** for each predator whose center is within **`R_CAPTURE_RANGE`** (if `CONTRIBUTOR_BONUS_ENABLED`)

## Observation Vector (Per Predator)

Type: fixed-size `np.float32` vector.

Current size:
- `OBS_SIZE = 64`

Layout:
1. Self (4):
   - own pos `(x, y)` normalized by `WORLD_SCALE`
   - own vel `(vx, vy)` normalized by `DRONE_SPEED`
2. Prey slot (6):
   - `prey_visible` flag (1 iff **team sensing** is active)
   - relative prey pos
   - relative prey vel
   - prey distance
   - **Team sensing:** if **any** predator is within `R_SENSE` of the prey, **all** predators get full prey-relative features (shared spotter). If **no** predator is in range, the slot is zeroed (no stale coordinates).
3. Teammates (`K_TEAMMATES=5`, each 6):
   - valid flag
   - relative pos
   - relative vel
   - distance
   - sorted by distance ascending, then padded
4. Obstacles (`M_OBSTACLES=4`, each 5):
   - valid flag
   - relative obstacle center
   - obstacle characteristic radius
   - distance
   - sorted by distance ascending, then padded
5. Borders (4):
   - distances to left/right/top/bottom

Sensing rule:
- **Prey:** at least one predator must be within `R_SENSE` of the prey; then **all** agents get prey-relative features (team broadcast). No stale prey coords when nobody senses prey.
- **Teammates / obstacles:** radius-only (`distance <= R_SENSE`) from self, as before.
- No line-of-sight test; obstacles do not block sensing.

## Action Spaces

### Core `Environment`
- `step` input: `dict[int, tuple[float, float]]`
- Meaning: desired world-frame velocity components
- Env clips to `DRONE_SPEED`

### PettingZoo `PursuitParallelEnv`
- Agents: `predator_0` ... `predator_7`
- `action_space(agent) = Box(low=-DRONE_SPEED, high=DRONE_SPEED, shape=(2,), dtype=float32)`
- `observation_space(agent) = Box(shape=(OBS_SIZE,), dtype=float32)`

## API Usage

### Core env (index-based)

```python
from swarm_env.environment import Environment

env = Environment(seed=0)
obs, infos = env.reset(seed=0)

actions = {i: (10.0, 0.0) for i in range(env.num_agents)}
obs, rewards, terminations, truncations, infos = env.step(actions)
```

`infos` includes `episode_state`, `tactical_state`, `step`, and **`capture`** (`CaptureStatus`: drones in ring, contributor indices, hold progress, wall count).

### PettingZoo parallel env (string-agent)

```python
from swarm_env.parallel_env import PursuitParallelEnv

env = PursuitParallelEnv(seed=0)
obs, infos = env.reset(seed=0)

while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
```

## AgileRL Integration Notes

Use `PursuitParallelEnv` as the training entrypoint.

Typical setup:
1. Instantiate `PursuitParallelEnv` (with `action_repeat` aligned across train/eval/demo — default **`4`** in `train.py`)
2. Call `reset()` for dict observations
3. Build policy networks from `action_space` / `observation_space`
4. Train with the shared reward signal (already handled by env)

Because this env follows PettingZoo Parallel API, it plugs into PettingZoo-compatible AgileRL pipelines directly.

Training script: `train.py` (check `--help` for curriculum, vectorized envs, fixed-seed evaluation, and other flags). The default curriculum now eases multiple task dimensions, not just prey speed: early stages can disable obstacles, make prey globally visible, and shorten the capture hold. For visual evaluation of a saved policy, use `demo.py` and keep `--action-repeat` consistent with training.

## Tests

Run from `env/`:

```bash
python -m pytest tests/ -q
```

Or individually:

```bash
python tests/test_capture.py
python tests/test_parallel_api.py
```

Coverage:
- Four drones in the ring, open arena → `CAPTURED` after hold
- Three drones only, open arena → never `CAPTURED`
- Three drones + one wall intersecting the blue circle → `CAPTURED` after hold
- Prey forced exit after `T_HIDE_MAX`
- PettingZoo `parallel_api_test` and random rollout smoke test

### Manual layout demo (`demo_manual_spawn_test.py`)

Prey starts at arena center with scripted policy effectively off (`prey_speed_factor=0`); **drones push the prey** via a demo-only `ManualPushDemoEnv` physics pass (overlap resolution + light damping). Three bots are stationary; one drone uses **WASD**. **Capture rules are the same as training** (`COMBO_CAPTURE_NEED`, wall + drone combo). Obstacle-free layout.

## File Map

```text
env/
├── main.py
├── demo.py
├── demo_manual_spawn_test.py
├── train.py
├── requirements.txt
├── tests/
│   ├── __init__.py
│   ├── test_capture.py
│   └── test_parallel_api.py
└── swarm_env/
    ├── __init__.py
    ├── arena.py
    ├── obstacle.py
    ├── drone.py
    ├── prey.py
    ├── capture.py
    ├── environment.py
    ├── parallel_env.py
    └── config.py
```

## Team Notes

- If you want exactly 4 learning predators, set `DRONE_COUNT = 4` in `config.py`.
- Agent names in `PursuitParallelEnv` are generated from `DRONE_COUNT`.
- **`capture.py`** owns the tactical FSM, distance capture hold, and helpers (`predators_in_capture_range`, `walls_intersecting_capture_circle`, `nearest_predator_distance`, `flee_angle_from_nearest_predator`).
- Checkpoints trained under **older angular capture + reward shaping** are not behaviorally or reward-matched to the current distance-only rules; retrain after changing capture semantics.
