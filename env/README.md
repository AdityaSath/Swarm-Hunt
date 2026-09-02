# V1 Multi-Agent Pursuit Environment

A 2D continuous multi-agent pursuit environment with:
- Learning agents: predator drones
- Non-learning prey: bouncing ball (wall reflection)
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

### Reliable scripted capture showcase (`showcase_demo.py`)

```bash
python showcase_demo.py
```

This runs a deterministic formation controller that predicts the moving prey
and assigns each predator a different surrounding slot. It uses the real core
physics, obstacles, and capture rules, but it is explicitly labeled as a
scripted baseline rather than a learned policy. The default `0.5x` prey speed
provides reliable captures across randomized layouts.

Controls are `Space` to pause, `R` to reset, and `Esc` to quit. Add
`--show-targets` to visualize formation assignments or `--no-obstacles` for an
open arena. Drone and prey positions are randomized on every reset. Pass
`--seed 7` to get a reproducible sequence of layouts (`7`, `8`, `9`, ...).

### Trained MATD3 demo (`demo.py`)

Runs pygame with policies loaded from a checkpoint (default: most recent `models/MATD3/*.pt`).

Controls: same as above (`Space`, `R`, `Esc`).

Important: training uses **`action_repeat=2`** on `PursuitParallelEnv` (see `train.py`). The demo repeats each policy action for the same number of physics steps so behavior matches training. Override only if your checkpoint used a different repeat.

CLI (common options):

| Flag | Default | Notes |
|------|---------|--------|
| `--checkpoint` | latest in `models/MATD3/` | Path to `.pt` |
| `--bc-checkpoint` | none | Preview an actor produced by `pretrain_bc.py` |
| `--action-repeat` | `2` | Physics steps per policy decision; match `train.py` |
| `--prey-speed-factor` | `1.0` | Scales prey speed in the env |
| `--prey-bounce-scale` | config default | Multiplier on bounce prey speed (`PREY_BOUNCE_SPEED_SCALE`) |
| `--episodes` | `0` | Auto-exit after N completed episodes (`0` = run until closed) |

Example:

```bash
python demo.py --checkpoint models/MATD3/MATD3_best.pt --action-repeat 2
```

## Current Implemented Defaults

Defined in `swarm_env/config.py`:
- Predators: `DRONE_COUNT = 6`
- Prey: `1`
- `r_prey = 2 * r_pred`
- `v_prey = v_pred` at full difficulty
- `R_SENSE = 8 * r_prey` (team prey sensing / teammate slots)
- `R_DANGER = 4 * r_prey` (FREE ↔ THREATENED, nearest-predator distance)
- `R_CAP = 2.5 * r_prey` (legacy base; capture ring is scaled from this)
- `R_CAPTURE_RANGE = 1.2 * R_CAP` (predators inside this radius count toward capture)
- `CAPTURE_HOLD_SECONDS = 2.0` → `CAPTURE_HOLD_STEPS = 2 * FPS` (consecutive steps the hold must stay valid)
- `COMBO_CAPTURE_NEED = 4` (need **walls intersecting blue circle + drones inside `R_CAPTURE_RANGE`** ≥ this to build the hold)
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
2. Physics and collisions (prey integrates and bounces on arena edges)
3. Distance capture check + tactical FSM
4. Reward computation
5. Observation assembly

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
- Acceleration clipping: actual velocity approaches the desired velocity by at
  most `DRONE_MAX_ACCELERATION * DT` per physics step. With the default
  `320 px/s²`, reaching full speed from rest takes 0.25 seconds.
- Integration: `position += velocity * DT`
- Predators cannot pass through walls, obstacles, or other predators

### Demo-mode movement
When `actions=None`, the env uses smooth random wandering:
- Keeps a sampled velocity for about 1 second
- Then samples a new direction/speed

For RL training, always pass explicit actions.

## How Prey Works (Bouncing Ball)

File: `swarm_env/prey.py`

- Spawned near the arena center with a random heading; speed is `PREY_SPEED * prey_speed_factor * PREY_BOUNCE_SPEED_SCALE` (see `config.py`).
- Not a learning agent: velocity changes only via arena **wall reflection** (`Arena.clamp_and_bounce` in `environment.py`).
- Prey passes through obstacles; only arena edges reflect.

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

## Rewards

The environment combines shared capture/chase terms with role-aware local shaping:

- shared terminal capture and timeout rewards;
- shared progress in mean predator–prey distance and capture-hold progress;
- per-agent progress toward its assigned formation slot;
- formation-slot proximity and team angular-coverage bonuses;
- a chase-direction bonus only while outside the capture ring;
- collision, idle, boundary, and edge-stall penalties.

The stable role objective allows a shared actor to learn different surrounding
behavior for each predator instead of sending all drones to the same point.

## Observation Vector (Per Predator)

Type: fixed-size `np.float32` vector.

Current size:
- `OBS_SIZE = 66`

Layout:
1. Self (4):
   - own pos `(x, y)` normalized by `WORLD_SCALE`
   - own vel `(vx, vy)` normalized by `DRONE_SPEED`
2. Formation role (2):
   - `sin(role_angle)`, `cos(role_angle)` for a stable per-agent ring slot
3. Prey slot (6):
   - `prey_visible` flag (currently always `1`)
   - relative prey pos
   - relative prey vel
   - prey distance
4. Teammates (`K_TEAMMATES=5`, each 6):
   - valid flag
   - relative pos
   - relative vel
   - distance
   - sorted by distance ascending, then padded
5. Obstacles (`M_OBSTACLES=4`, each 5):
   - valid flag
   - relative obstacle center
   - obstacle characteristic radius
   - distance
   - sorted by distance ascending, then padded
6. Borders (4):
   - distances to left/right/top/bottom

Teammates and obstacles use radius-only sensing (`distance <= R_SENSE`). The
prey state is globally available so learning focuses on coordination and
capture rather than exploration.

## Action Spaces

### Core `Environment`
- `step` input: `dict[int, tuple[float, float]]`
- Meaning: desired world-frame velocity components
- Env clips to `DRONE_SPEED`

### PettingZoo `PursuitParallelEnv`
- Agents: `predator_0` ... `predator_5` (when `DRONE_COUNT = 6`)
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

## Learning Pipeline

The recommended path is expert imitation followed by MATD3 fine-tuning.

1. Collect randomized scripted demonstrations:

   ```bash
   python collect_expert.py --episodes 500 --output data/expert_capture.h5
   ```

2. Behavior-clone the shared actor:

   ```bash
   python pretrain_bc.py \
     --dataset data/expert_capture.h5 \
     --output models/BC/formation_actor.pt \
     --epochs 30
   ```

3. Measure the cloned policy before RL:

   ```bash
   python evaluate.py \
     --bc-checkpoint models/BC/formation_actor.pt \
     --episodes 100 --prey-speed-factor 0.5
   ```

4. Fine-tune with the five-stage curriculum:

   ```bash
   python train.py \
     --bc-checkpoint models/BC/formation_actor.pt \
     --max-steps 2000000 --num-envs 4
   ```

   The curriculum progresses from a stationary prey/open arena to the full
   moving-prey obstacle task. Captures and timeouts are both terminal replay
   transitions. `models/MATD3/MATD3_best.pt` is selected by held-out capture
   rate rather than training reward.

5. Evaluate and display the learned policy:

   ```bash
   python evaluate.py --checkpoint models/MATD3/MATD3_best.pt --episodes 100
   python demo.py --checkpoint models/MATD3/MATD3_best.pt
   ```

Training, evaluation, and the visual demo all default to `action_repeat=2`.
Changing the observation layout, movement physics, or action repeat invalidates
older checkpoints.

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
- PettingZoo `parallel_api_test` and random rollout smoke test

### Manual layout demo (`demo_manual_spawn_test.py`)

Prey starts at arena center with zero velocity (`prey_speed_factor=0`); **drones push the prey** via a demo-only `ManualPushDemoEnv` physics pass (overlap resolution + light damping). Three bots are stationary; one drone uses **WASD**. **Capture rules are the same as training** (`COMBO_CAPTURE_NEED`, wall + drone combo). Obstacle-free layout.

## File Map

```text
env/
├── main.py
├── showcase_demo.py
├── demo.py
├── demo_manual_spawn_test.py
├── collect_expert.py
├── pretrain_bc.py
├── evaluate.py
├── swarm_ml.py
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
    ├── formation_controller.py
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
