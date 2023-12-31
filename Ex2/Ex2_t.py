import robot
from enum import Enum
import random
import sys 
import robot 
import _utils
import numpy as np
from time import sleep
import time 
import random

# Create a robot object and initialize
arlo = robot.Robot()

class Direction(Enum):
    LEFT = 'left'
    RIGHT = 'right'

def sensor():
    front = arlo.read_front_ping_sensor()
    back = arlo.read_back_ping_sensor()
    right = arlo.read_right_ping_sensor()
    left = arlo.read_left_ping_sensor()
    
    return front, back, right, left

def drive(front_threshold=350, back_threshold = 350, side_threshold=250, sharp_turn_angle=180.0, slight_turn_angle=45.0, ninety_turn_angle=90.0):
    # Define conditions

    conditions = [
        # All sensors detect obstacles
        (lambda f, b, l, r: f < front_threshold and b < back_threshold and l < side_threshold and r < side_threshold, lambda: arlo.stop()),

        # Three sensors detect obstacles
        (lambda f, b, l, r: f < front_threshold and b < back_threshold and l < side_threshold, lambda: _utils.sharp_turn(Direction.RIGHT, ninety_turn_angle)),
        (lambda f, b, l, r: f < front_threshold and b < back_threshold and r < side_threshold, lambda: _utils.sharp_turn(Direction.LEFT, ninety_turn_angle)),
        (lambda f, b, l, r: f < front_threshold and l < side_threshold and r < side_threshold, lambda: _utils.sharp_turn(Direction.LEFT, sharp_turn_angle)),
        (lambda f, b, l, r: b < back_threshold and l < side_threshold and r < side_threshold, lambda: _utils.sharp_turn(Direction.RIGHT, sharp_turn_angle)),

        # Two sensors detect obstacles
        (lambda f, b, l, r: f < front_threshold and b < back_threshold, lambda: _utils.sharp_turn(Direction.LEFT, sharp_turn_angle)),
        (lambda f, b, l, r: f < front_threshold and l < side_threshold, lambda: _utils.sharp_turn(Direction.RIGHT, ninety_turn_angle)),
        (lambda f, b, l, r: f < front_threshold and r < side_threshold, lambda: _utils.sharp_turn(Direction.LEFT, ninety_turn_angle)),
        (lambda f, b, l, r: b < back_threshold and l < side_threshold, lambda: _utils.sharp_turn(Direction.RIGHT, ninety_turn_angle)),
        (lambda f, b, l, r: b < back_threshold and r < side_threshold, lambda: _utils.sharp_turn(Direction.LEFT, ninety_turn_angle)),
        (lambda f, b, l, r: l < side_threshold and r < side_threshold, lambda: _utils.sharp_turn(Direction.LEFT, sharp_turn_angle)),

        # One sensor detects an obstacle
        (lambda f, b, l, r: f < front_threshold, lambda: _utils.sharp_turn(random.choice(list(Direction)), ninety_turn_angle)),
        (lambda f, b, l, r: b < back_threshold, lambda: _utils.sharp_turn(random.choice(list(Direction)), ninety_turn_angle)),
        (lambda f, b, l, r: l < side_threshold, lambda: _utils.sharp_turn(Direction.RIGHT, slight_turn_angle)),
        (lambda f, b, l, r: r < side_threshold, lambda: _utils.sharp_turn(Direction.LEFT, slight_turn_angle))
    ]

    while True:
        pingFront, pingBack, pingRight, pingLeft = sensor()
        arlo.go_diff(50, 50, 1, 1)  

        # Check conditions and execute corresponding actions
        for condition, action in conditions:
            if condition(pingFront, pingBack, pingRight, pingLeft):
                action()
                break
        drive()
        
# Start driving
drive()
