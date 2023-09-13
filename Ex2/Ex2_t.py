import robot
from enum import Enum
import random

# Create a robot object and initialize
arlo = robot.Robot()

class Direction(Enum):
    LEFT = 1
    RIGHT = 2

def sensor():
    front = arlo.read_front_ping_sensor()
    back = arlo.read_back_ping_sensor()
    right = arlo.read_right_ping_sensor()
    left = arlo.read_left_ping_sensor()
    
    return front, back, right, left

def sharp_turn(direction, angle):
    if direction == Direction.LEFT:
        arlo.go_diff(50, 50, 0, 1)  # Assuming this is how you turn the robot left
    else:
        arlo.go_diff(50, 50, 1, 0)  # Assuming this is how you turn the robot right
    arlo.sleep(angle/90)  # Assuming the robot turns 90 degrees in 1 second

def drive(front_threshold=350, side_threshold=250, sharp_turn_angle=180.0, slight_turn_angle=45.0):
    # Define conditions
    conditions = [
        (lambda f, l, r: f < front_threshold and l < side_threshold and r < side_threshold, lambda: sharp_turn(Direction.LEFT, sharp_turn_angle)),
        (lambda f, l, r: f < front_threshold and l < side_threshold, lambda: sharp_turn(Direction.RIGHT, 90.0)),
        (lambda f, l, r: f < front_threshold and r < side_threshold, lambda: sharp_turn(Direction.LEFT, 90.0)),
        (lambda f, l, r: l < side_threshold and r < side_threshold, lambda: sharp_turn(Direction.LEFT, sharp_turn_angle)),
        (lambda f, l, r: f <= front_threshold, lambda: sharp_turn(random.choice(list(Direction)), 90.0)),
        (lambda f, l, r: l < side_threshold, lambda: sharp_turn(Direction.RIGHT, slight_turn_angle)),
        (lambda f, l, r: r < side_threshold, lambda: sharp_turn(Direction.LEFT, slight_turn_angle))
    ]

    while True:
        pingFront, pingBack, pingRight, pingLeft = sensor()
        arlo.go_diff(50, 50, 1, 1)  # Assuming this is how you move the robot forward

        # Check conditions and execute corresponding actions
        for condition, action in conditions:
            if condition(pingFront, pingLeft, pingRight):
                action()
                break  # Exit once an action is taken

# Start driving
drive()
