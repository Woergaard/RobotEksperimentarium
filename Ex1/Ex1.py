from time import sleep
import robot
import time

# Create a robot object and initialize
arlo = robot.Robot()

"""
Using the robot object make code that does a simple movement and motion calibration:
1.  Write a program that moves
the robot around in a square.
2. Continuous motion: Write a program that moves the robot continuously
(without stopping), e.g. in a figure 8 path.
3. Motion accuracy estimation: Measure the drift of your robot.
"""

rightWheelFactor = 1.0
leftWheelFactor = 0.95

oneMeterSeconds = 2.5
ninetyDegreeTurnSeconds = 1.5


# Move forward
#arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
#sleep(2)
#arlo.stop()


# NON BLOCKING KØRSEL FIRKANT

def right_turn(start, ninetyDegreeTurnSeconds):
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 0)
    isTurning = True
    while (isTurning): # or some other form of loop
        if (time.perf_counter() - start > ninetyDegreeTurnSeconds): #stop after 5 seconds
            arlo.stop()
            isTurning = False

def drive_forward(start, oneMeterSeconds, meters):
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
    isDriving = True
    while (isDriving): # or some other form of loop
        if (time.perf_counter() - start > oneMeterSeconds * meters): #drive x meters
            arlo.stop()
            isDriving = False

def move_in_square_non_blocking(meters):
    start = time.perf_counter()

    for _ in range(3):
        drive_forward(start, oneMeterSeconds, meters)

        start = time.perf_counter()

        right_turn(start, ninetyDegreeTurnSeconds)

        start = time.perf_counter()

meters = 1.0
move_in_square_non_blocking(meters)

"""
def move_in_square():
    # Move forward
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
    sleep(2)
    
    # Turn right
    arlo.go_diff(leftWheelFactor*50, -rightWheelFactor*50, 1, 1)
    sleep(1)
    
    # Move forward
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
    sleep(2)
    
    # Turn right
    arlo.go_diff(leftWheelFactor*50, -rightWheelFactor*50, 1, 1)
    sleep(1)
    
    # Move forward
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
    sleep(2)
    
    # Turn right
    arlo.go_diff(leftWheelFactor*50, -rightWheelFactor*50, 1, 1)
    sleep(1)
    
    # Move forward
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
    sleep(2)
    
    # Stop
    arlo.stop()


def move_in_figure_eight():
    while True:
        # Move forward
        arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
        sleep(2)
        
        # Turn right
        arlo.go_diff(leftWheelFactor*50, -rightWheelFactor*50, 1, 1)
        sleep(1)
        
        # Move forward
        arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
        sleep(2)
        
        # Turn left
        arlo.go_diff(-leftWheelFactor*50, rightWheelFactor*50, 1, 1)
        sleep(1)


def measure_drift():
    # Read the initial encoder counts
    initial_left_encoder_counts = arlo.read_left_wheel_encoder()
    initial_right_encoder_counts = arlo.read_right_wheel_encoder()

    # Move the robot forward for a certain amount of time
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
    sleep(2)  # Adjust the sleep time as needed
    arlo.stop()

    # Read the final encoder counts
    final_left_encoder_counts = arlo.read_left_wheel_encoder()
    final_right_encoder_counts = arlo.read_right_wheel_encoder()

    # Calculate the distances traveled by each wheel
    left_distance = final_left_encoder_counts - initial_left_encoder_counts
    right_distance = final_right_encoder_counts - initial_right_encoder_counts

    # Calculate and return the drift
    drift = abs(left_distance - right_distance)
    return drift




move_in_square()
#move_in_figure_eight()
#measure_drift()
"""


