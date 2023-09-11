
import robot 
import _utils
from time import sleep
import time 

# Create a robot object and initialize
arlo = robot.Robot()

# SOFIE 
def sensorValues():
        forwards = arlo.read_front_ping_sensor()
        backwards = arlo.read_back_ping_sensor()
        left = arlo.read_left_ping_sensor()
        right = arlo.read_right_ping_sensor()

        return forwards, backwards, left, right

def go_and_sensor(seconds):
    '''
    Funktionen venter i den tid, som robotten skal udføre en bestemt action.
    Samme tanke som Sleep, men kan måske sørge for den kan holde sig opdateret på sine omgivelser.
    Argumenter:
        start:    starttidspunkts
        seconds:  sekunder, der skal ventes
    '''
    start = time.perf_counter()
    isDriving = True
    while (isDriving):
        if (time.perf_counter() - start > seconds):
            isDriving = False
        forwards, backwards, left, right = sensorValues()
        sensorLst = [forwards, backwards, left, right]
        sleep(0.041)
        sensorBools = [x > 200 for x in sensorLst]
        if sensorBools[0]:
            isDriving = False
            return sensorBools
        
def drive_anywhere(direction):
    '''
    Funktionen kører robotten et antal meter.
    Argumenter:
        start:    starttidspunkt
        direction:  enten 'forwards', 'backwards', 'left' eller 'right'
        meters:   meter, der skal køres
    '''

    if direction == 'right' or 'left':
        _utils.sharp_turn(direction, 90)

    if direction == 'forwards':
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)
    elif direction == 'backwards':
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 0)


def drive_forward_and_sense(meters):

    drive_anywhere('forward', meters)
    
    driveSeconds = _utils.metersToSeconds(meters)

    sensorBools = go_and_sensor(driveSeconds)







    if 

    if random.choice(sensorBools[1:2]) == True


    arlo.stop()



# sensorNames = ['forwards', 'backwards', 'left', 'right']





        sleep(0.041)
        for i in sensorLst < 200:
            isDriving = False
            return i, sensorLst 

# SLUT SOFIE






####### VIDERE ARBEJDE PÅ CAROLINES ########

def drive():
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



# right og left sensor afstand, kan go
'''
### GAMLE ####
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
'''      
            
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