"""PyTest for blocked-move capture detection.

Places the prey at center and positions all predators evenly on a small
circle so they form a surround. The environment should declare capture
within a few steps.
"""

import math
import sys
import os

import pygame

# allow imports from parent ``env`` package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swarm_env.environment import Environment
from swarm_env.capture import PreyTacticalState
from swarm_env.config import R_CAP


def test_blocked_move_capture_surround():
    env = Environment()
    # place prey at center
    px = env._width / 2
    py = env._height / 2
    env.prey.position.x = px
    env.prey.position.y = py

    # place predators evenly on a tight circle so they are contributors
    dist = 0.5 * R_CAP
    n = len(env.drones)
    for i in range(n):
        d = env.drones[i]
        ang = 2 * math.pi * i / n
        d.position.x = px + dist * math.cos(ang)
        d.position.y = py + dist * math.sin(ang)
        d.velocity.x = 0
        d.velocity.y = 0

    captured = False
    max_steps = 40
    for t in range(max_steps):
        _, _, terms, truncs, info = env.step(None)
        tactical = info.get("tactical_state")
        if getattr(tactical, "name", None) == "CAPTURED":
            captured = True
            break

    assert captured, "Prey should be captured by a tight surround"
