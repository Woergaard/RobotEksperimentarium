
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
