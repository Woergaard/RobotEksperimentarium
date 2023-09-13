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
def sensor():
    front = arlo.read_front_ping_sensor()
    back = arlo.read_back_ping_sensor()
    right = arlo.read_right_ping_sensor()
    left = arlo.read_left_ping_sensor()
    
    return front, back, right, left

def drive():
    pingFront, pingLeft, pingRight, pingBack = sensor()
    arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
    dirLst = ['right', 'left']
    while (pingFront > 450 and pingLeft > 350 and pingRight > 350):
        pingFront, pingLeft, pingRight, pingBack = sensor()
        print(pingLeft)
    if (pingFront < 450 and pingLeft < 350 and pingRight < 350):
        _utils.sharp_turn('left', 180.0)    
    elif (pingFront < 450 and pingLeft < 350):
        _utils.sharp_turn('right', 90.0)
    elif (pingFront < 450 and pingRight < 350):
        _utils.sharp_turn('left', 90.0)
    elif (pingLeft < 350 and pingRight < 350):
        _utils.sharp_turn('left', 180.0)
    elif(pingFront <= 450):
        randomDirection = random.choice(dirLst)
        _utils.sharp_turn(randomDirection, 90.0)
    elif (pingLeft < 350): 
        _utils.sharp_turn('right', 45.0)
    elif (pingRight < 350):
        _utils.sharp_turn('left', 45.0)
    
        
    drive()


drive()




########################
#      Exercise 2      #
########################
