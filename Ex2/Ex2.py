import sys
import robot 
import _utils
import numpy as np


#from library import robot, _utils
from time import sleep
import time 

# Create a robot object and initialize
arlo = robot.Robot()

################################
#          Exercise 1          #
################################
# CAROLINE
def sensor():
    front = arlo.read_front_ping_sensor()
    back = arlo.read_back_ping_sensor()
    right = arlo.read_right_ping_sensor()
    left = arlo.read_left_ping_sensor()
    
    return front, back, right, left
Wheel = 47.2 

def drive(wheel):
    arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
    isDriving = True
    while (isDriving): # or some other form of loop
        pingFront = arlo.read_front_ping_sensor() 
        pingLeft = arlo.read_left_ping_sensor()
        pingRight = arlo.read_right_ping_sensor()
        pingBack = arlo.read_back_ping_sensor()
        
        if (pingFront <= 200): 
            _utils.sharp_turn('right', _utils.degreeToSeconds(90.0))
            turnSeconds = _utils.degreeToSeconds(90.0) # kan man bare bruge _utils.nintydegree... som defineret i utils?
            _utils.go(turnSeconds)
            arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
            arlo.stop()

            if (pingRight <= 200):
                arlo.stop() 
                # Turn left

                if(pingLeft <= 200): 
                    arlo.stop()
                    # Turn back
                        
                    if(pingBack <= 200):
                        arlo.stop()
                        # STOP STOP STOP 
                

drive(Wheel)

#SLUT CAROLINE 



"""
def drive_forward(start, oneMeterSeconds, meters):
    #arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
    isDriving = True
    while (isDriving): # or some other form of loop
        sensorReadings()
        if (time.perf_counter() - start > oneMeterSeconds * meters): #drive x meters
            isDriving = False

start = time.perf_counter()
oneMeterSeconds = 10
meters = 1.0

drive_forward(start, oneMeterSeconds, meters)

#drive_forward()
arlo.stop()

"""





################################
#          Exercise 2          #
################################

# ASGER 

def sonar_calibration():
     # arlo.read_front_ping_sensor()
     # arlo.read_back_ping_sensor()
     # arlo.read_right_ping_sensor()
     # arlo.read_left_ping_sensor()
    
    while(readings))

    return 

# SLUT ASGER