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
#Wheel = 47.2 

def drive():#wheel):
    arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
    isDriving = True
    while (isDriving): # or some other form of loop
        pingFront = arlo.read_front_ping_sensor() 
        pingLeft = arlo.read_left_ping_sensor()
        pingRight = arlo.read_right_ping_sensor()
        pingBack = arlo.read_back_ping_sensor()
        
        if (pingFront <= 200): 
            _utils.sharp_turn('right', _utils.degreeToSeconds(90.0))
            turnSeconds = _utils.degreeToSeconds(90.0) 
            _utils.go(turnSeconds)
            arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
            _utils.metersToSeconds(2.8)
            arlo.stop()
            _utils.sharp_turn('left', _utils.degreeToSeconds(90.0))
            turnSeconds = _utils.degreeToSeconds(90.0) 
            _utils.go(turnSeconds)
            arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)

            if (pingRight and pingFront<= 200):
                _utils.sharp_turn('left', _utils.degreeToSeconds(90.0))
                turnSeconds = _utils.degreeToSeconds(90.0) 
                _utils.go(turnSeconds)
                arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
                arlo.stop() 
                # Turn left 
                

                if(pingLeft and pingFront and pingRight <= 200): 
                    _utils.move_around_own_axis('left', 180) 
                    arlo.stop()
                    # Turn back
                        
                    if(pingBack <= 200):
                        print("I am stuck")
                        arlo.stop()
                        # STOP STOP STOP 
                

drive()

#SLUT CAROLINE 

### tilføjelse af tilfældighed af h/v for undgå loop
### tilføj at den kigger til h/v hele tiden for at se, hvornår den er fri

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
'''
def sonar_calibration(): 

    # Record the front sensor's measurements

    readings = np.zeros((5,5))
    
    for i in range(5):
        for j in range(5):
            readings[i,j] = arlo.read_front_ping_sensor()

        # Gives time to change distance

        sleep(5)

        
    # Standard deviations of the measurements
    estimate_dist = np.mean(readings, axis = 0)
    np.std(estimate_dist)
        
    return 
'''
# SLUT ASGER