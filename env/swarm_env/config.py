"""Tunable constants for the swarm environment."""

ARENA_WIDTH = 800
ARENA_HEIGHT = 600

DRONE_COUNT = 3
DRONE_RADIUS = 15
DRONE_SPEED = 80.0

# Vision cone (degrees); half-angle from center, so 45 = 90-degree total cone
DRONE_VISION_ANGLE_DEG = 45
DRONE_VISION_RANGE = 150

# Sensor/perception: max distance drone can detect obstacles and other drones
DRONE_PERCEPTION_RANGE = 120

OBSTACLE_SIZES = {
    "small": (30, 30),
    "large": (80, 60),
}

# Predefined obstacle positions (x, y, size_type) - center of obstacle
# size_type: "small" or "large"
OBSTACLE_POSITIONS = [
    (150, 150, "small"),
    (400, 300, "small"),
    (650, 200, "small"),
    (250, 450, "large"),
    (550, 100, "large"),
    (100, 350, "small"),
    (700, 450, "small"),
    (350, 100, "large"),
    (500, 450, "large"),
    (200, 280, "small"),
]

FPS = 60
DT = 1.0 / FPS
