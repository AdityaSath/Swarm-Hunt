"""Tunable constants for the V1 pursuit environment."""

import math

# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------
ARENA_WIDTH = 1200
ARENA_HEIGHT = 900
WORLD_SCALE = max(ARENA_WIDTH, ARENA_HEIGHT)  # normalization denominator

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
FPS = 60
DT = 1.0 / FPS  # simulation timestep (single source of truth)
MAX_STEPS = 30 * FPS  # episode truncation: 30 s wall-clock at nominal FPS

# ---------------------------------------------------------------------------
# Predators
# ---------------------------------------------------------------------------
DRONE_COUNT = 8
DRONE_RADIUS = 15
# Increased predator max speed to make pursuit more effective.
# Previously 80.0 — raised to 120.0 so predators can close on prey faster.
DRONE_SPEED = 120.0  # v_pred — max speed, clips desired-velocity magnitude

# ---------------------------------------------------------------------------
# Prey
# ---------------------------------------------------------------------------
PREY_RADIUS = 2 * DRONE_RADIUS         # 30
# Keep prey somewhat slower than predators so predators can catch up.
# Set prey speed relative to predator speed (80% of predator speed).
PREY_SPEED = 0.8 * DRONE_SPEED         # (v_prey)

# ---------------------------------------------------------------------------
# Sensing (radius-only, no LOS, obstacles do not block)
# ---------------------------------------------------------------------------
R_SENSE = 8 * PREY_RADIUS              # 240

# ---------------------------------------------------------------------------
# Capture geometry (predators + borders only; obstacles excluded in V1)
# ---------------------------------------------------------------------------
R_CAP = 3.0 * PREY_RADIUS              # 90   capture contribution radius
R_WALL_CAP = 1.5 * PREY_RADIUS         # 45   border counts as blocker when prey is this close
# Terminal capture threshold: largest gap must be < this (degrees)
# Raised from 70° to 90° to make the terminal containment condition easier
# to reach during early tuning runs.
PHI_ESCAPE_MAX = math.radians(90)
# Minimum predators within R_CAP for capture. Keep this aligned with the
# capture tests: 4 contributors can capture; 3 contributors must not.
MIN_PREDATOR_CONTRIBUTORS = 4
# Consecutive steps the capture condition must hold before declaring capture.
# Lowering from 5 -> 2 speeds up captures during debugging.
T_HOLD = 2

# ---------------------------------------------------------------------------
# Tactical FSM thresholds (with hysteresis margins)
# ---------------------------------------------------------------------------
R_DANGER = 4 * PREY_RADIUS             # 120  FREE → THREATENED when any predator enters
PHI_CONTAINED = math.radians(110)       # THREATENED → CONTAINED when gap < this
MARGIN_CONTAINED = math.radians(15)     # leave CONTAINED only when gap > PHI_CONTAINED + this
MARGIN_THREATENED = 1.5 * PREY_RADIUS  # leave THREATENED only when nearest predator > R_DANGER + this

# ---------------------------------------------------------------------------
# Prey hiding
# ---------------------------------------------------------------------------
T_HIDE_MAX = 20  # max steps prey may stay inside an obstacle

# ---------------------------------------------------------------------------
# Demo / behavior tuning
# ---------------------------------------------------------------------------
# Weights used by the demo scripted policy to blend pursuit vs flanking vs inertia.
PURSUIT_WEIGHT = 0.82
FLANK_WEIGHT = 0.18
INERTIA_WEIGHT = 0.08

# Multiplier for flank radius relative to capture radius / prey size.
# Reduced from 1.2 to 0.9 so flank targets lie inside or near the capture radius
# (predators will approach closer to the prey instead of staying outside it).
FLANK_RADIUS_MULT = 0.9

# If True, use a dynamic gap-seeking algorithm to pick flank angles that
# target the largest empty angular sectors around the prey. Otherwise,
# predators use evenly-spaced flank positions. Dynamic mode can sometimes
# leave large escape gaps when predators start clustered; for stability
# prefer evenly-spaced flanking during early experiments.
ENABLE_DYNAMIC_FLANK = False

# Number of candidate sample angles used when computing dynamic flank targets.
# Higher values give finer-grained gap selection at slight CPU cost.
FLANK_ANGLE_CANDIDATES = 72

# ---------------------------------------------------------------------------
# Grid search / discovery behavior
# ---------------------------------------------------------------------------
GRID_ROWS = 3
GRID_COLS = 3
GRID_TARGET_JITTER = 0.25            # fraction of cell half-size used for random search targets
GRID_SEARCH_SPEED_FRAC = 0.75
GRID_CONVERGE_SPEED_FRAC = 1.0
GRID_TARGET_REACHED_DIST = 45.0

# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------
K_TEAMMATES = 5  # max teammate slots per predator observation
M_OBSTACLES = 4  # max obstacle slots per predator observation

# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
REWARD_CAPTURE = 10.0
REWARD_TIMEOUT = -5.0
REWARD_THREATENED = 0.5
REWARD_CONTAINED = 1.5
REWARD_CONTAINMENT_STEP = 0.05
REWARD_ESCAPE = -1.0               # prey escapes from CONTAINED → FREE
PENALTY_OBSTACLE_COLLISION = -0.5
PENALTY_PREDATOR_COLLISION = -0.2
PENALTY_IDLE = -0.01               # per step when speed < IDLE_SPEED_THRESHOLD
IDLE_SPEED_THRESHOLD = 1.0

DIST_SHAPING_CLIP = 0.1            # per-step distance-shaping clipped to [-clip, +clip]
CONTRIBUTOR_BONUS = 0.02           # tiny per-step bonus for predators within R_CAP
CONTRIBUTOR_BONUS_ENABLED = True

# Additional shaping / per-agent reward coefficients to encourage pursuit
# and flanking behavior for early training and debugging.
# Per-agent distance shaping coefficient (multiplied by clipped delta/world_scale)
PURSUIT_REWARD_COEF = 0.5
# Clip for per-agent distance shaping (absolute value, unitless after dividing by WORLD_SCALE)
AGENT_DIST_SHAPING_CLIP = 0.05
# Extra per-step reward for predators recognized as contributors (flanking)
FLANK_REWARD = 0.08
# Reward the first visit to a grid cell by any predator.
GRID_DISCOVERY_REWARD = 0.18
# Team reward for broad search coverage across the 3x3 grid.
GRID_COVERAGE_REWARD = 0.03
# Reward a predator that lands in the prey's current grid cell.
PREY_GRID_DISCOVERY_REWARD = 0.6
# Reward agents for moving toward the currently discovered/remembered prey grid.
GRID_CONVERGE_REWARD_COEF = 0.25
GRID_CONVERGE_SHAPING_CLIP = 0.05
# Extra reward when contributors approach from separated angular sectors.
FLANK_DIVERSITY_REWARD = 0.15

# Dispersion phase: at episode start encourage agents to spread out.
# Number of steps (frames) during which dispersion shaping is active.
DISPERSION_PHASE_STEPS = 5 * FPS
# Per-agent dispersion coefficient (applied to increase in nearest-neighbor distance)
DISPERSION_REWARD_COEF = 0.08
# Clip for dispersion shaping after normalizing by WORLD_SCALE
DISPERSION_SHAPING_CLIP = 0.03
# Early search reward for maintaining real separation, not only increasing it.
INITIAL_SEPARATION_REWARD_COEF = 0.15
INITIAL_SEPARATION_TARGET_DIST = 0.75 * min(ARENA_WIDTH / GRID_COLS, ARENA_HEIGHT / GRID_ROWS)
# Early search reward for moving away from the nearest teammate.
AWAY_FROM_TEAMMATE_REWARD_COEF = 0.08
# Early search reward for occupying different grid cells.
INITIAL_UNIQUE_GRID_REWARD = 0.12
# Extra early reward for searching edge/corner cells instead of collapsing to center.
INITIAL_OUTER_GRID_REWARD = 0.10
# Early penalty for remaining in the center grid cell before prey discovery.
INITIAL_CENTER_GRID_PENALTY = -0.12

# See-prey incentives: when agents detect the prey (within R_SENSE) give
# an immediate reward to encourage detection and then a team bonus when
# a configurable majority of agents can see the prey.
SEE_PREY_REWARD = 0.5
SEE_PREY_TEAM_BONUS = 2.0
# Fraction of agents that must see prey to qualify for the team bonus
SEE_PREY_TEAM_FRAC = 0.5

# Capture redefinition: radius around the prey we test for "blocked" moves.
# If the prey cannot move by CAP_BLOCK_DIST in any sampled direction without
# colliding with a predator, it is considered captured.
CAP_BLOCK_DIST = 1.5 * PREY_RADIUS
CAP_BLOCK_ANGLE_SAMPLES = 36

# ---------------------------------------------------------------------------
# Obstacles (unchanged layout)
# ---------------------------------------------------------------------------
OBSTACLE_SIZES = {
    "small": (39, 39),
    "large": (104, 78),
}

OBSTACLE_POSITIONS = [
    (225, 225, "large"),
    (600, 450, "large"),
    (375, 675, "large"),
    (825, 150, "large"),
    (150, 525, "large"),
    (1050, 675, "large"),
    (580, 150, "large"),
    (750, 675, "large"),
    (150, 750, "large"),
    (1050, 500, "large"),
]
