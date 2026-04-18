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
# Capture — distance-only ring (no angular gap logic)
# ---------------------------------------------------------------------------
R_CAP = 2.5 * PREY_RADIUS              # 75   base; capture ring is 1.2× this
R_CAPTURE_RANGE = 1.2 * R_CAP          # 90   predators within this radius count toward capture
CAPTURE_HOLD_SECONDS = 2.0
CAPTURE_HOLD_STEPS = int(CAPTURE_HOLD_SECONDS * FPS)
# Capture hold: (walls intersecting blue circle) + (drones inside R_CAPTURE_RANGE) >= this
COMBO_CAPTURE_NEED = 4

# ---------------------------------------------------------------------------
# Tactical FSM (threat proximity only; capture is distance hold above)
# ---------------------------------------------------------------------------
R_DANGER = 4 * PREY_RADIUS             # 120  FREE → THREATENED when any predator enters
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

# Rewards
# ---------------------------------------------------------------------------
REWARD_CAPTURE_TEAM = 5.0
REWARD_CAPTURE_CONTRIBUTOR = 5.0
REWARD_TIMEOUT = -2.0
REWARD_THREATENED = 0.25
REWARD_COMBINED_PROGRESS = 0.15
REWARD_HOLD_PROGRESS = 0.02
REWARD_CONTAINED = 0.0
REWARD_CONTAINMENT_STEP = 0.0
REWARD_ESCAPE = 0.0                # prey escapes from CONTAINED → FREE
PENALTY_OBSTACLE_COLLISION = -0.25
PENALTY_PREDATOR_COLLISION = -0.10
PENALTY_IDLE = 0.0                 # per step when speed < IDLE_SPEED_THRESHOLD
IDLE_SPEED_THRESHOLD = 1.0

DIST_SHAPING_CLIP = 0.005          # per-step distance-shaping clipped to [-clip, +clip]
ENCIRCLEMENT_SHAPING_SCALE = 0.0   # weight for gap-closing shaping (stronger than distance)
ENCIRCLEMENT_SHAPING_CLIP = 0.0    # per-step encirclement delta clamp
CONTRIBUTOR_BONUS = 0.0            # tiny per-step bonus for predators within R_CAP
CONTRIBUTOR_BONUS_ENABLED = False

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
