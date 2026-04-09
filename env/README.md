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
- `R_SENSE = 8 * r_prey`
- `R_DANGER = 4 * r_prey`
- `R_CAP = 2.5 * r_prey`
- `R_WALL_CAP = 1.5 * r_prey`
- `phi_escape_max = 70 deg`
- `T_HOLD = 5`
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
4. Capture geometry + tactical FSM
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
1. If threatened: move toward largest escape gap
2. If threatened and obstacle nearby: may enter obstacle
3. If hidden: remain hidden up to `T_HIDE_MAX`, then forced exit
4. Otherwise: move away from predator cluster direction

Physics rule:
- Prey can pass through obstacles
- Prey is clamped by arena borders

## Capture Logic (Border-Aware Angular Enclosure)

File: `swarm_env/capture.py`

One geometry implementation is reused by both:
- terminal capture checks
- prey escape-gap steering

V1 blockers:
- Predator blockers: predators within `R_CAP`
- Border blockers: walls when prey is within `R_WALL_CAP`
- Obstacles are excluded from capture geometry

Terminal capture condition:
- largest escape gap `< PHI_ESCAPE_MAX`
- predator contributors `>= MIN_PREDATOR_CONTRIBUTORS` (currently 4)
- condition holds for `T_HOLD` consecutive steps

## Tactical State Machine

Prey tactical states:
- `FREE`
- `THREATENED`
- `CONTAINED`
- `CAPTURED`

Hysteresis:
- Enter `CONTAINED` when `gap < PHI_CONTAINED`
- Leave `CONTAINED` when `gap > PHI_CONTAINED + MARGIN_CONTAINED`
- Threatened recovery uses `MARGIN_THREATENED`

Global episode state:
- `IN_PURSUIT`
- `CAPTURED`
- `TIMEOUT`

## Rewards (Shared Team Reward)

All predators receive the same base team reward per step.

Included terms:
- Terminal:
  - capture: `+10`
  - timeout: `-5`
- Transitions:
  - `FREE -> THREATENED`: `+0.5`
  - `THREATENED -> CONTAINED`: `+1.5`
  - escape from containment: `-1.0`
- Maintenance:
  - containment step: `+0.05`
- Penalties:
  - obstacle collision: `-0.5` (shared contribution)
  - predator collision: `-0.2` (shared contribution)
  - idle penalty: small per-step
- Shaping:
  - mean predator-prey distance delta, clipped by `DIST_SHAPING_CLIP`
- Optional:
  - tiny contributor bonus for actual capture contributors

## Observation Vector (Per Predator)

Type: fixed-size `np.float32` vector.

Current size:
- `OBS_SIZE = 64`

Layout:
1. Self (4):
   - own pos `(x, y)` normalized by `WORLD_SCALE`
   - own vel `(vx, vy)` normalized by `DRONE_SPEED`
2. Prey slot (6):
   - `prey_visible` flag
   - relative prey pos
   - relative prey vel
   - prey distance
   - zeroed if prey not in `R_SENSE`
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
- Radius-only (`distance <= R_SENSE`)
- No line-of-sight test
- Obstacles do not block sensing

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

Training script: `train.py` (check `--help` for curriculum, vectorized envs, and other flags). For visual evaluation of a saved policy, use `demo.py` and keep `--action-repeat` consistent with training.

## Tests

Run from `env/`:

```bash
python tests/test_capture.py
python tests/test_parallel_api.py
```

Coverage:
- 4 contributors + border-aware enclosure captures after `T_HOLD`
- 3 contributors does not capture
- Prey forced exit after `T_HIDE_MAX`
- PettingZoo `parallel_api_test` and random rollout smoke test

## File Map

```text
env/
├── main.py
├── demo.py
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
- Keep `capture.py` as the single source of truth for escape-gap geometry.
