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
# Goal during this training pass: teach drones to CHASE the prey.
# Polish penalties (boundary / idle / collisions) are softened or zeroed so
# the per-step pursuit signal dominates the gradient.
REWARD_CAPTURE = 14.0
REWARD_TIMEOUT = -5.0
REWARD_THREATENED = 0.85
PENALTY_OBSTACLE_COLLISION = -0.05
PENALTY_PREDATOR_COLLISION = -0.02
PENALTY_IDLE = 0.0
IDLE_SPEED_THRESHOLD = 0.5

# Per-agent dense potential shaping: every step, reward a small *negative* of the
# normalized distance to prey. Net effect: drones gain reward by being closer,
# every single step, without telescoping.  Tunable scale.
DIST_POTENTIAL_WEIGHT = 0.038

# Legacy delta-distance shaping is OFF (replaced by potential above). Kept for
# reference; weight 0 disables it.
DIST_SHAPING_CLIP = 0.12
PER_AGENT_DIST_SHAPING_WEIGHT = 0.0

# Dense bonus for moving toward prey (velocity vs unit vector prey - predator)
REWARD_VELOCITY_TOWARD_PREY = 0.11
VELOCITY_TOWARD_MIN_DIST = 20.0  # skip when essentially on top of prey (avoids noise)

# Stronger pursuit right after spawn (episode step < this at 60 FPS)
CHASE_BOOTSTRAP_STEPS = 300
CHASE_BOOTSTRAP_MULT = 3.1

# When inside the capture contribution radius: reward staying + prefer slow speed so
# drones can hold the ring without overshooting / oscillating out.
REWARD_IN_CAPTURE_RING_PER_STEP = 0.028
REWARD_SLOW_IN_RING = 0.085

# Boundary penalty disabled while learning to chase (re-enable after pursuit works)
BOUNDARY_MARGIN_PENALTY = 0.0
PENALTY_BOUNDARY_PROXIMITY = 0.0

# Edge x straggler: penalize border hugging when farther from prey than the team median.
# e = in-edge-band strength, s = how much farther than median (clamped). Penalty = w*e*s.
EDGE_STRAGGLER_BAND_PX = 95.0
PENALTY_EDGE_STRAGGLER = 0.07
STRAGGLER_DIST_SCALE = 220.0

CONTRIBUTOR_BONUS = 0.065
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
