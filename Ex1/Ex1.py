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
leftWheelFactor = 1.06225 # mellem 1.02 og 1.0625, alt efter hvor meget den har kørt


oneMeterSeconds = 2.75 #Den tager måske et sekund fra når den skal starte op
ninetyDegreeTurnSeconds = 0.95 #Den drejer for lang tid, vi har ikke testet nuværdende værdi
circleTurnSecond = ninetyDegreeTurnSeconds*4 # 4 skal hyperparametertybes #
startCircleTurnSeconds = 5.65  #Caroline leger, tidnen for at kører 360 grader (starte og slutte samme sted)
circleTurnSeconds = 5.20 

# Move forward
#arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
#sleep(oneMeterSeconds)
#arlo.stop()


# NON BLOCKING KØRSEL FIRKANT SubEx1

def right_turn(start, turnSeconds):
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 0)
    isTurning = True
    while (isTurning): # or some other form of loop
        if (time.perf_counter() - start > turnSeconds): #stop after 5 seconds
            isTurning = False

def drive_forward(start, oneMeterSeconds, meters):
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
    isDriving = True
    while (isDriving): # or some other form of loop
        if (time.perf_counter() - start > oneMeterSeconds * meters): #drive x meters
            isDriving = False

def move_in_square_non_blocking(meters, i):
    start = time.perf_counter()

    for _ in range(i):
        for _ in range(4):
            drive_forward(start, oneMeterSeconds, meters)

            start = time.perf_counter()

            right_turn(start, ninetyDegreeTurnSeconds)

            start = time.perf_counter()

meters = 1.0
move_in_square_non_blocking(meters, 3)


### NONBLOCKING KØRSEL OTTEKANT SubEx2

def left_turn(start, turnSeconds):
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 0, 1)
    isTurning = True
    while (isTurning):
        if (time.perf_counter() - start > turnSeconds):
            isTurning = False

def move_in_figure_eight_non_blocking_sofie():
    start = time.perf_counter()
    
    for _ in range(1):
        right_turn(start, circleTurnSecond)
        
        start = time.perf_counter()
        
        left_turn(start, circleTurnSecond)

        start = time.perf_counter()

# KØRSEL

meters = 1.0

#move_in_figure_eight_non_blocking_sofie()

#arlo.stop()

## CAROLINES NOTER HERUNDE SLUT
def circle_right_turn(start, circleTurnSeconds, meters):
    arlo.go_diff(leftWheelFactor*100, rightWheelFactor*40, 1, 1) #har ændret i wheel faktoren, bare en start værdi (ikke fast)
    isTurning = True
    while(isTurning): 
        if(time.perf_counter() - start > circleTurnSeconds * meters):
            isTurning = False
            
def circle_left_turn(start, circleTurnSeconds, meters):
    arlo.go_diff(leftWheelFactor*40, rightWheelFactor*100, 1, 1)
    isTurning = True
    while(isTurning):
            if(time.perf_counter() - start > circleTurnSeconds * meters):
                isTurning = False

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

meters = 1.0

#move_in_figure_eight_non_blocking(meters, 1)

arlo.stop()