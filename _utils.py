from time import sleep
import robot
import time

### PARAMETERUDREGNING ###

def degreeToSeconds(degrees):
    ninetyDegreeTurnSeconds = 0.95
    return (ninetyDegreeTurnSeconds/90) * degrees

def metersToSeconds(meters):
    oneMeterSeconds = 2.8
    return oneMeterSeconds * meters

### BEVÆGELSE ###

def go(start, seconds):
    isDriving = True
    while (isDriving):
        if (time.perf_counter() - start > seconds):
            isDriving = False

def drive_forward(start, meters):
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)

    driveSeconds = metersToSeconds(meters)
    go(start, driveSeconds)

def sharp_turn(start, direction, degrees):
    if direction == "right":
        arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 0)
    elif direction == "left":
        arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 0, 1)
    
    turnSeconds = degreeToSeconds(degrees)
    go(start, turnSeconds)

def soft_turn(start, direction, degrees):
    if direction == "right":
        arlo.go_diff(leftWheelFactor*100, rightWheelFactor*40, 1, 1)
    elif direction == "left":
        arlo.go_diff(leftWheelFactor*40, rightWheelFactor*100, 1, 1)

    turnSeconds = degreeToSeconds(degrees)
    go(start, turnSeconds)
            
### SAMMENSAT BEVÆGELSE ###

def move_in_square_non_blocking(meters):
    start = time.perf_counter()

    for _ in range(4):
        drive_forward(start, 1.0)

        start = time.perf_counter()

        sharp_turn(start, 'right', 90)
    
        start = time.perf_counter()

def move_around_own_axis():
    start = time.perf_counter()
    
    for _ in range(1):
        sharp_turn(start, 'right', 90)
        
        start = time.perf_counter()
        
        sharp_turn(start, 'left', 90)

        start = time.perf_counter()

def move_in_figure_eight_non_blocking(meters, i):
    start = time.perf_counter()
    
    circle_right_turn(start, startCircleTurnSeconds, meters)
    
    start = time.perf_counter()
    
    circle_left_turn(start, circleTurnSeconds, meters)

    start = time.perf_counter()

    for _ in range(i):   
            circle_right_turn(start, startCircleTurnSeconds, meters)
            
            start = time.perf_counter()
            
            circle_left_turn(start, circleTurnSeconds, meters)

            start = time.perf_counter()



# Create a robot object and initialize
arlo = robot.Robot()

rightWheelFactor = 1.0
leftWheelFactor = 1.045


oneMeterSeconds = 2.80 #Den tager måske et sekund fra når den skal starte op
ninetyDegreeTurnSeconds = 0.95#95 #Den drejer for lang tid, vi har ikke testet nuværdende værdi
circleTurnSecond = ninetyDegreeTurnSeconds*4 # 4 skal hyperparametertybes #
startCircleTurnSeconds = 5.65  #Caroline leger, tidnen for at kører 360 grader (starte og slutte samme sted)
circleTurnSeconds = 5.20 
