"""Tunable constants for the V1 pursuit environment."""

# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------
ARENA_WIDTH = 1400
ARENA_HEIGHT = 800
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
# Maximum change in velocity per second.  At this value a stationary drone
# reaches full speed in 0.25 s, while a full-speed reversal takes 0.5 s.
# Keeping this in the shared physics (rather than smoothing only the renderer)
# makes training and visual evaluation behave identically.
DRONE_MAX_ACCELERATION = 320.0

# ---------------------------------------------------------------------------
# Prey
# ---------------------------------------------------------------------------
PREY_RADIUS = 2 * DRONE_RADIUS         # 30
# Full difficulty = same top speed as the predators.
PREY_SPEED = DRONE_SPEED
# Bouncing prey speed = PREY_SPEED * prey_speed_factor * this (wall reflections in arena)
PREY_BOUNCE_SPEED_SCALE = 1.0

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

# Stable per-agent formation roles. Agent i owns one evenly spaced point on
# this ring, exposed as sin/cos(role_angle) in its observation.
FORMATION_TARGET_RADIUS = 0.72 * R_CAPTURE_RANGE

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
# Goal during this training pass: teach drones to CHASE the prey.
# Make the team reward progress and capture, then use local penalties to stop
# drifting, edge camping, and obstacle pinning.
REWARD_CAPTURE = 18.0
REWARD_TIMEOUT = -6.0
REWARD_THREATENED = 1.0
PENALTY_OBSTACLE_COLLISION = -0.08
PENALTY_PREDATOR_COLLISION = -0.04
PENALTY_IDLE = -0.015
IDLE_SPEED_THRESHOLD = 4.0

# Shared chase shaping: reward the team for reducing mean prey distance.
TEAM_MEAN_PROGRESS_CLIP = 0.015
TEAM_MEAN_PROGRESS_WEIGHT = 22.0
TEAM_CAPTURE_RANGE_WEIGHT = 0.16
TEAM_HOLD_PROGRESS_WEIGHT = 0.10

# Role-aware shaping: reward progress toward the assigned formation slot and
# reward team angular coverage so agents learn to surround rather than bunch.
FORMATION_PROGRESS_CLIP = 0.015
FORMATION_PROGRESS_WEIGHT = 18.0
FORMATION_PROXIMITY_RADIUS = 180.0
FORMATION_PROXIMITY_REWARD = 0.04
ANGULAR_COVERAGE_REWARD = 0.05

# Signed pursuit reward: toward-prey velocity is positive, away-from-prey is negative.
REWARD_VELOCITY_TOWARD_PREY = 0.18
VELOCITY_TOWARD_MIN_DIST = 20.0  # skip when essentially on top of prey (avoids noise)

# Stronger pursuit right after spawn while the shared policy is still orienting.
CHASE_BOOTSTRAP_STEPS = 240
CHASE_BOOTSTRAP_MULT = 2.2

# When inside the capture contribution radius: reward staying + prefer slow speed so
# drones can hold the ring without overshooting / oscillating out.
REWARD_IN_CAPTURE_RING_PER_STEP = 0.03
REWARD_SLOW_IN_RING = 0.03

# Re-enable wall pressure so corners are never a stable solution.
BOUNDARY_MARGIN_PENALTY = 85.0
PENALTY_BOUNDARY_PROXIMITY = -0.08

# Edge x straggler: penalize border hugging when farther from prey than the team median.
# e = in-edge-band strength, s = how much farther than median (clamped). Penalty = w*e*s.
EDGE_STRAGGLER_BAND_PX = 115.0
PENALTY_EDGE_STRAGGLER = 0.18
STRAGGLER_DIST_SCALE = 180.0

# Explicit anti-stall penalty near the edges/corners.
STUCK_EDGE_MARGIN = 28.0
STUCK_SPEED_THRESHOLD = 3.0
STUCK_STEPS = 30
PENALTY_STUCK = -0.18

# Preserve tangential motion on contact so drones slide out instead of pinning.
WALL_TANGENT_DAMPING = 1.0
OBSTACLE_TANGENT_DAMPING = 0.92
PREDATOR_TANGENT_DAMPING = 0.88

CONTRIBUTOR_BONUS = 0.0
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
