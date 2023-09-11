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

def drive( ):
    arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
    while (pingFront > 300 and pingLeft > 200 and pingRight > 200):
        pingFront, pingLeft, pingRight, pingBack = sensor()
    # nu er der sket noget!!!
    ## indsæt håndtering
    if(pingFront <= 300):
        randomDirection = np.random('right', 'left')
        _utils.sharp_turn(randomDirection, 90.0)
    elif (pingFront < 300 and pingLeft < 200):
        _utils.sharp_turn('right', 90.0)
    elif (pingFront < 300 and pingRight < 200):
        _utils.sharp_turn('left', 90.0)
    elif (pingLeft < 200):
        _utils.sharp_turn('right', 45)
    elif (pingRight < 200):
        _utils.sharp_turn('left', 45)
    elif (pingLeft < 200 and pingRight < 200):
        _utils.sharp_turn('left', 180)
    elif (pingFront < 300 and pingLeft < 200 and pingRight < 200):
        _utils.sharp_turn('left', 180)        
    drive()




# right og left sensor afstand, kan go


#### GAMLE ####
#Wheel = 47.2 

def left_turn_return_route(pingFront, pingLeft, pingRight, pingBack):
    start = time.perf_counter()
    arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
    while (pingLeft <= 300 and pingFront > 200):
            pingFront, pingLeft, pingRight, pingBack = sensor()
    if (pingLeft > 300):    
        sleep(2.0)
        timeDriven = time.perf_counter() - start
        _utils.sharp_turn('left', 90.0)
    else:
        print('stuck')
    return timeDriven


def turn_right_back_to_route(pingFrong, pingLeft, pingRight, pingBack, timeDriven):
    arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
    sleep(timeDriven)
    _utils.sharp_turn('right', 90.0)


def drive(): #wheel):
    arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
    isDriving = True
    while (isDriving): # or some other form of loop
        pingFront, pingLeft, pingRight, pingBack = sensor()
        
        if (pingFront <= 200): 
            _utils.sharp_turn('right', 90.0)
            
            timeDriven = left_turn_return_route(pingFront, pingLeft, pingRight, pingBack)
            _ = left_turn_return_route(pingFront, pingLeft, pingRight, pingBack)
            
            turn_right_back_to_route(timeDriven)
            drive() # nu er robotten back on route
            
    
drive()        
                
            
# 1. Kør frem indtil pingFront <= 200
# 2. Drej til højre (90 grader)
# 3. Kør frem indtil pingLeft > 300 + 1 sekund mere 
# 4. Drej til venstre (90 grader)
# 5. Kør frem indtil  pingLeft > 300 + 1 sekund mere 
# 6. Drej til venstre (90 grader)
# 7. Kør frem indtil  pingLeft > 300 + 1 sekund mere 
# 8. Drej til højre (90 grader)
# 9. Kør fremad i et sekund 




            #turnSeconds = _utils.degreeToSeconds(90.0) 
            #_utils.go(turnSeconds)
           
            #sleep(_utils.metersToSeconds(2.8))
            #arlo.stop()#
#            _utils.sharp_turn('left', 90.0)
            #turnSeconds = _utils.degreeToSeconds(90.0) 
            #_utils.go(turnSeconds)
#            arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
            #sleep(_utils.metersToSeconds(2.8))
            #arlo.stop()

"""
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
"""              



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