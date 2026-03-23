"""Tunable constants for the swarm environment."""

import math

ARENA_WIDTH = 1200   # 800 * 1.5
ARENA_HEIGHT = 900   # 600 * 1.5

DRONE_COUNT = 10
DRONE_RADIUS = 15
DRONE_SPEED = 80.0

# Helicopter-style movement: thrust (forward/back) + steer (turn left/right)
# thrust in [-1, 1]: -1=full backward, 0=no forward/back, 1=full forward
# steer in [-1, 1]: -1=turn left, 0=no turn, 1=turn right
# Hold thrust+steer = curved path (circle). Steer only = spin in place.
DRONE_MAX_TURN_RATE = math.pi / 3  # rad/s at full steer

# Sensor/perception: max distance drone can detect obstacles and other drones
DRONE_PERCEPTION_RANGE = 120

# Obstacle repulsion: soft push away when near (1/distance^2 falloff)
OBSTACLE_REPULSION_RANGE = 100
OBSTACLE_REPULSION_STRENGTH = 15000

OBSTACLE_SIZES = {
    "small": (39, 39),    # 30 * 1.3
    "large": (104, 78),   # 80 * 1.3, 60 * 1.3
}

# Predefined obstacle positions (x, y, size_type) - center of obstacle
# size_type: "small" or "large" (scaled 1.5x with arena)
OBSTACLE_POSITIONS = [
    (225, 225, "large"),
    (600, 450, "large"),
    (375, 675, "large"),
    (825, 150, "large"),
    (150, 525, "large"),
    (1050, 675, "large"),
    (580, 150, "large"),   # moved to avoid overlap with 450,150
    (750, 675, "large"),
    (150, 750, "large"),
    (1050, 500, "large"),
]

FPS = 60
DT = 1.0 / FPS
