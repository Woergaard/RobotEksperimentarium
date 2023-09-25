from time import sleep
import robot
import time
import random

### DEFINEREDE PARAMETRE ###
arlo = robot.Robot()

rightWheelFactor = 1.0
leftWheelFactor = 1.06225
standardSpeed = 50.0

### PARAMETERUDREGNING ###

def degreeToSeconds(degrees):
    '''
    Funktionen omregner grader, der skal drejes, til sekunder, der skal ventes.
    Argumenter:
        degrees:    grader, der skal drejes
    '''
    ninetyDegreeTurnSeconds = 0.95 
    return (ninetyDegreeTurnSeconds/90) * degrees

def metersToSeconds(meters):
    '''
    Funktionen omregner meter, der skal køres, til sekunder, der skal ventes.
    Argumenter:
        meters:    meter, der skal køres
    '''
    oneMeterSeconds = 2.8
    return oneMeterSeconds * meters

### BEVÆGELSE ###

def wait(seconds):
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

def drive(direction, meters):
    '''
    Funktionen kører robotten et antal meter.
    Argumenter:
        start:    starttidspunkt
        direction:  enten 'forwards', 'backwards', 'left' eller 'right'
        meters:   meter, der skal køres
    '''
    if direction == 'forwards':
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)
    elif direction == 'backwards':
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 0)

    driveSeconds = metersToSeconds(meters)
    wait(driveSeconds)

def sharp_turn(direction, degrees):
    '''
    Funktionen kører robotten et antal meter.
    Argumenter:
        start:    starttidspunkt
        direction:  enten 'right' eller 'left'
        degrees:   grader, der skal køres
    '''
    if direction == "right":
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 0)
    elif direction == "left":
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 1)
    
    turnSeconds = degreeToSeconds(degrees)
    wait(turnSeconds)

def soft_turn(direction, degrees):
    '''
    Funktionen kører robotten et antal meter.
    Argumenter:
        start:    starttidspunkt
        direction:  enten 'right' eller 'left'
        degrees:   grader, der skal køres
    '''
    if direction == "right":
        arlo.go_diff(leftWheelFactor*100, rightWheelFactor*40, 1, 1)
    elif direction == "left":
        arlo.go_diff(leftWheelFactor*40, rightWheelFactor*100, 1, 1)

    turnSeconds = degreeToSeconds(degrees)
    wait(turnSeconds)
            
### SAMMENSAT BEVÆGELSE ###

def move_in_square(meters):
    for _ in range(4):
        drive(1.0)

        sharp_turn('right', 90)
    

def move_around_own_axis(i): 
    for _ in range(i):
        sharp_turn('right', 360)
                
        sharp_turn('left', 360)


def move_in_figure_eight(i):
    soft_turn('right', 360)

    soft_turn('left', 360)

    for _ in range(i):   
            soft_turn('right', 360)
            
            soft_turn('left', 360)



### SENSORERING ###

def sensor():
    front = arlo.read_front_ping_sensor()
    right = arlo.read_right_ping_sensor()
    left = arlo.read_left_ping_sensor()
    back = arlo.read_back_ping_sensor()
    
    return front, left, right, back

def turn_and_sense(direction):
    '''
    Funktionen drejer om egen akse, indtil der er frit.
    Argumenter:
        direction:  enten 'right' eller 'left'
    '''
    if direction == "right":
        r, l = 1, 0
    elif direction == "left":
        r, l = 0, 1
    else:
        print("Indtast enten 'left' eller 'right' som retning")
        return

    arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, r, l)
    pingFront, pingLeft, pingRight, pingBack = sensor()

    while (pingFront < 500 and pingLeft < 400 and pingRight < 400):
        pingFront, pingLeft, pingRight, pingBack = sensor()


def drive_and_sense():
    pingFront, pingLeft, pingRight, pingBack = sensor()

    if pingFront < 100:
        sharp_turn('left', 90.0)
    else:
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)
    
    dirLst = ['right', 'left']

    while (pingFront > 350 and pingLeft > 250 and pingRight > 250):
        pingFront, pingLeft, pingRight, pingBack = sensor()
        print(pingFront, pingLeft, pingRight, pingBack)
        sleep(0.041)
    
    # three sensors detected
    if (pingFront < 350 and pingLeft < 250 and pingRight < 250):
        turn_and_sense('left')  

    # two sensors detected
    elif (pingFront < 350 and pingLeft < 250):
        turn_and_sense('right')
    elif (pingFront < 350 and pingRight < 250):
        turn_and_sense('left')
    elif (pingLeft < 250 and pingRight < 250):
        turn_and_sense('left')

    # one sensors deteced
    elif(pingFront <= 350):
        randomDirection = random.choice(dirLst)
        turn_and_sense(randomDirection)
    elif (pingLeft < 250): 
        turn_and_sense('right')
    elif (pingRight < 250):
        turn_and_sense('left')
    
        
    drive_and_sense()


### MÅLING ###

def sensor_measure(): 
    while True:
        pingFront, pingBack, pingRight, pingLeft = sensor()
        print('front', pingFront, 'left', pingLeft, 'right', pingRight, 'back', pingBack)


###################################
######### ÆLDRE VERSIONER #########
###################################

### MANUELLE SENSORISKE KØRSLER ###
def drive_old():
    pingFront, pingLeft, pingRight, pingBack = sensor()
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
    dirLst = ['right', 'left']

    while (pingFront > 350 and pingLeft > 250 and pingRight > 250):
        pingFront, pingLeft, pingRight, pingBack = sensor()
        print(pingFront, pingLeft, pingRight, pingBack)
        sleep(0.041)
    
    # three sensors detected
    if (pingFront < 350 and pingLeft < 250 and pingRight < 250):
        sharp_turn('left', 180.0)  

    # two sensors detected
    elif (pingFront < 350 and pingLeft < 250):
        sharp_turn('right', 90.0)
    elif (pingFront < 350 and pingRight < 250):
        sharp_turn('left', 90.0)
    elif (pingLeft < 250 and pingRight < 250):
        sharp_turn('left', 180.0)

    # one sensors deteced
    elif(pingFront <= 350):
        randomDirection = random.choice(dirLst)
        sharp_turn(randomDirection, 90.0)
    elif (pingLeft < 250): 
        sharp_turn('right', 45.0)
    elif (pingRight < 250):
        sharp_turn('left', 45.0)
    
        
    drive_old()
