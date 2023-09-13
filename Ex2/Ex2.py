import sys 
import robot 
import _utils
import numpy as np


#from library import robot, _utils
from time import sleep
import time 
import random

# Create a robot object and initialize
arlo = robot.Robot()

################################
#          Exercise 1          #
################################
# CAROLINE
def sensor_ex():
    front = arlo.read_front_ping_sensor()
    back = arlo.read_back_ping_sensor()
    right = arlo.read_right_ping_sensor()
    left = arlo.read_left_ping_sensor()
    
    return front, back, right, left

def drive_ex():
    pingFront, pingBack, pingRight, pingLeft = sensor_ex()
    arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
    dirLst = ['right', 'left']
    # 
    while (pingFront > 350 and pingLeft > 250 and pingRight > 250):
        pingFront, pingBack, pingRight, pingLeft = sensor_ex()
        print('front', pingFront, 'left', pingLeft, 'right', pingRight, 'back', pingBack)
        sleep(0.041)
    if (pingFront < 350 and pingLeft < 250 and pingRight < 250):
        _utils.sharp_turn('left', 180.0)    
    elif (pingFront < 350 and pingLeft < 250):
        _utils.sharp_turn('right', 90.0)
    elif (pingFront < 350 and pingRight < 250):
        _utils.sharp_turn('left', 90.0)
    elif (pingLeft < 250 and pingRight < 250):
        _utils.sharp_turn('left', 180.0)
    elif(pingFront <= 350):
        randomDirection = random.choice(dirLst)
        _utils.sharp_turn(randomDirection, 90.0)
    elif (pingLeft < 250): 
        _utils.sharp_turn('right', 45.0)
    elif (pingRight < 250):
        _utils.sharp_turn('left', 45.0)
    
        
    #drive_ex()


#drive_ex()

def sensor_measure(): 
    while True:
        pingFront, pingBack, pingRight, pingLeft = sensor_ex()
        print('front', pingFront, 'left', pingLeft, 'right', pingRight, 'back', pingBack)

#sensor_measure()

_utils.drive_and_sense()

########################
#      Exercise 2      #
########################

