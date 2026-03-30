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
DRONE_SPEED = 80.0  # v_pred — max speed, clips desired-velocity magnitude

# ---------------------------------------------------------------------------
# Prey
# ---------------------------------------------------------------------------
PREY_RADIUS = 2 * DRONE_RADIUS         # 30
PREY_SPEED = 1.5 * DRONE_SPEED         # 120  (v_prey)

# ---------------------------------------------------------------------------
# Sensing (radius-only, no LOS, obstacles do not block)
# ---------------------------------------------------------------------------
R_SENSE = 8 * PREY_RADIUS              # 240

# ---------------------------------------------------------------------------
# Capture geometry (predators + borders only; obstacles excluded in V1)
# ---------------------------------------------------------------------------
R_CAP = 2.5 * PREY_RADIUS              # 75   capture contribution radius
R_WALL_CAP = 1.5 * PREY_RADIUS         # 45   border counts as blocker when prey is this close
PHI_ESCAPE_MAX = math.radians(70)       # terminal capture: largest gap must be < this
MIN_PREDATOR_CONTRIBUTORS = 4          # minimum predators within R_CAP for capture
T_HOLD = 5                              # consecutive steps the capture condition must hold

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
