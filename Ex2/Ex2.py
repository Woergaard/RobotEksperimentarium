import robot 
import _utils
from time import sleep
import time 

# Create a robot object and initialize
arlo = robot.Robot()


# Move forward
#arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)

def sensorReadings():
        # request to read Front sonar ping sensor
        print("Front sensor = ", arlo.read_front_ping_sensor())
        sleep(0.041)

        # request to read Back sonar ping sensor
        print("Back sensor = ", arlo.read_back_ping_sensor())
        sleep(0.041)

        # request to read Right sonar ping sensorutils

        print("Right sensor = ", arlo.read_right_ping_sensor())
        sleep(0.041)

        # request to read Left sonar ping sensor
        print("Left sensor = ", arlo.read_left_ping_sensor())
        sleep(0.041)



# CAROLINE
def sensor():
    foran = arlo.read_front_ping_sensor()
    bagved = arlo.read_back_ping_sensor()
    højre = arlo.read_right_ping_sensor()
    venstre = arlo.read_left_ping_sensor()
    
    return foran, bagved, højre, venstre
Wheel = 47.2 

def kør(wheel):
    leftWheelFactor = 0.3
    rightWheelFactor = 0.3
    
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
    isDriving = True
    while (isDriving): # or some other form of loop
        if (arlo.read_front_ping_sensor() <= 200 ): 
            arlo.stop

kør(Wheel)

#SLUT CAROLINE 



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
        for i in sensorLst < 200:
            isDriving = False
            return i, sensorLst 

def drive_forward_and_sense():
    arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)
    
    driveSeconds = _utils.metersToSeconds(meters)

    i, sensorLst = go_and_sensor(driveSeconds)

    if sensorLst[i]
    sensorNames = ['forwards', 'backwards', 'left', 'right']



# SLUT SOFIE



# ASGER

# SLUT ASGER


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







