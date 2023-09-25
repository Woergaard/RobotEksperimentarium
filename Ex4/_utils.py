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


### BRUG AF KAMERA ###
import cv2 # Import the OpenCV library
import time
from pprint import *
import numpy as np
import robot
from numpy import linalg

arlo = robot.Robot()

try:
    import picamera2
    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)

print("OpenCV version = " + cv2.__version__)

def turn_and_watch(direction, img):
    '''
    Funktionen drejer om egen akse, indtil den har fundet et landmark OG der er frit.
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

    arlo.go_diff(leftWheelFactor*standardSpeed*0.6, rightWheelFactor*standardSpeed*0.6, r, l)
    
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_corners, _, _ = cv2.aruco.detectMarkers(img, arucoDict)
    # If at least one marker is detected
    if len(aruco_corners) > 0:
        for corner in aruco_corners:
            # The corners are ordered as top-left, top-right, bottom-right, bottom-left
            top_left = corner[0][0]
            top_right = corner[0][1]
            bottom_right = corner[0][2]
            bottom_left = corner[0][3]

            print('Landmark detected')
            print(top_left, top_right, bottom_right, bottom_left)

        return True
    else: 
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 1)
        turnSeconds = degreeToSeconds(20)
        wait(turnSeconds)
        arlo.stop()
        wait(2.0)

        return False

def drive_to_landmarks(landmarks_lst):
    landmark = landmarks_lst[0]

    dist = landmark[0]
    angle = landmark[1]
    direction = landmark[2]
    id = landmark[3]
    tvec = landmark[4]

    print('Driving towards landmark ' + str(id) + ', distance = ' + str(dist))

    print('tvec = ' + str(tvec))

    sharp_turn(direction, angle)

    drive('forwards', (dist-150)/1000)

    print('Arrived at landmark!')
    arlo.stop()
    return
    

def pose_estimation(img, arucoDict): 
    """Funktionen returnerer en liste af placeringer i kameraets koordinatsystem samt id på de givne QR-koder"""
    """Denne funktion skal finde positionen af et landmark i forhold til robotten og udregne vinklen og afstanden til landmarket."""

    # 1. Udregne afstanden til QR code - DONE 
    # 2. Udregn vinkel til QR code 
    # 3. Dreje robotten, så den er vinkelret på QR code
    # 4. Køre fremad, indtil robotten er tæt nok på QR code

    aruco_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, arucoDict)
    w, h = 1280, 720
    focal_length = 1744.36 
    camera_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]])
    arucoMarkerLength = 145.0
    distortion = 0
    
    # Giver distancen til boxen
    _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(aruco_corners, arucoMarkerLength, camera_matrix, distortion)
    
    lst = []
    
    for i in range(len(ids)):
        world_angle = np.arccos(np.dot(tvecs[i]/np.linalg.norm(tvecs[i]), np.array([0, 0, 1])))
        print(world_angle)
        if np.dot(tvecs[i], np.array([1, 0, 0])) < 0:
            direction = 'left'
        else:
            direction = 'right'
        print(direction)

        lst.append([linalg.norm(tvecs[i]), world_angle, direction, ids[i][0], tvecs[i]])
    
    lst.sort()
    
    namelst = ['distance', 'vinkel', 'retning', 'id', 'tvec']    

    for element in lst:
        for i in range(len(element)):
            print(namelst[i] + '=' + str(element[i]) + '\n') 

    return lst

def camera():
    # Open a camera device for capturing
    imageSize = (1280, 720)
    FPS = 60
    cam = picamera2.Picamera2()
    frame_duration_limit = int(1/FPS * 1000000) # Microseconds
    # Change configuration to set resolution, framerate
    picam2_config = cam.create_video_configuration({"size": imageSize, "format": 'RGB888'},
                                                                controls={"FrameDurationLimits": (frame_duration_limit, frame_duration_limit)},
                                                                queue=False)
    cam.configure(picam2_config) # Not really necessary
    cam.start(show_preview=False)

    print(cam.camera_configuration()) # Print the camera configuration in use

    time.sleep(1)  # wait for camera to setup

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # Open a window
    WIN_RF = "Example 1"
    cv2.namedWindow(WIN_RF)
    cv2.moveWindow(WIN_RF, 100, 100)
    
    while cv2.waitKey(4) == -1: # Wait for a key pressed event
        image = cam.capture_array("main")
        
        # Show frames
        cv2.imshow(WIN_RF, image)
        #landmark_drive('left', image, arucoDict)
        if turn_and_watch('left', image) == True: 
            arlo.stop()
            landmarks_lst = pose_estimation(image, arucoDict)
            drive_to_landmarks(landmarks_lst)
            #arlo.stop()
            time.sleep(15)


