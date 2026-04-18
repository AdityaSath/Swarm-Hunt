"""Tunable constants for the V1 pursuit environment."""

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
DRONE_COUNT = 6
DRONE_RADIUS = 15
DRONE_SPEED = 80.0  # v_pred — max speed, clips desired-velocity magnitude

# ---------------------------------------------------------------------------
# Prey
# ---------------------------------------------------------------------------
PREY_RADIUS = 2 * DRONE_RADIUS         # 30
# Slightly faster than predators so pursuit stays challenging but physically reachable
PREY_SPEED = 1.05 * DRONE_SPEED
# Bouncing prey speed = PREY_SPEED * prey_speed_factor * this (wall reflections in arena)
PREY_BOUNCE_SPEED_SCALE = 0.2

# ---------------------------------------------------------------------------
# Sensing (radius-only, no LOS, obstacles do not block)
# ---------------------------------------------------------------------------
R_SENSE = 8 * PREY_RADIUS              # 240

# ---------------------------------------------------------------------------
# Capture — distance-only ring (no angular gap logic)
# ---------------------------------------------------------------------------
R_CAP = 2.5 * PREY_RADIUS              # 75   base; capture ring is 1.2× this
R_CAPTURE_RANGE = 1.2 * R_CAP          # 90   predators within this radius count toward capture
CAPTURE_HOLD_SECONDS = 2.0
CAPTURE_HOLD_STEPS = int(CAPTURE_HOLD_SECONDS * FPS)
# Capture hold: (walls intersecting blue circle) + (drones inside R_CAPTURE_RANGE) >= this
COMBO_CAPTURE_NEED = 3

# ---------------------------------------------------------------------------
# Tactical FSM (threat proximity only; capture is distance hold above)
# ---------------------------------------------------------------------------
R_DANGER = 4 * PREY_RADIUS             # 120  FREE → THREATENED when any predator enters
MARGIN_THREATENED = 1.5 * PREY_RADIUS  # leave THREATENED only when nearest predator > R_DANGER + this

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
PENALTY_OBSTACLE_COLLISION = -0.35
PENALTY_PREDATOR_COLLISION = -0.15
PENALTY_IDLE = -0.004
IDLE_SPEED_THRESHOLD = 0.5

# Distance shaping (per predator only; no team-average term)
DIST_SHAPING_CLIP = 0.12
PER_AGENT_DIST_SHAPING_WEIGHT = 1.0

# Dense bonus for moving toward prey (velocity vs unit vector prey - predator)
REWARD_VELOCITY_TOWARD_PREY = 0.018
VELOCITY_TOWARD_MIN_DIST = 20.0  # skip when essentially on top of prey (avoids noise)

# Soft penalty for hugging arena walls (reduces corner camping)
BOUNDARY_MARGIN_PENALTY = 48.0
PENALTY_BOUNDARY_PROXIMITY = -0.018

CONTRIBUTOR_BONUS = 0.04
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
