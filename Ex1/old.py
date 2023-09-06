
"""
def circle_right_turn(start, circleTurnSeconds, meters):
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*40, 1, 1) #har ændret i wheel faktoren, bare en start værdi (ikke fast)
    isTurning = True
    while(isTurning): 
        if(time.perf_counter() - start > circleTurnSeconds * meters): 
            arlo.stop()
            isTurning = False
            
def circle_left_turn(start, circleTurnSeconds, meters):
    arlo.go_diff(leftWheelFactor*40, rightWheelFactor*50, 1, 1)
    isTurning = True
    while(isTurning):
            if(time.perf_counter() - start > circleTurnSeconds * meters):
                arlo.stop()
                isTurning = False

def move_in_figure_eight_non_blocking(meters):
    start = time.perf_counter()
    
    for _ in range(1):
        circle_right_turn(start, circleTurnSeconds, meters)
        
        start = time.perf_counter()
        
        circle_left_turn(start, circleTurnSeconds, meters)

        start = time.perf_counter()

meters = 1.0



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


