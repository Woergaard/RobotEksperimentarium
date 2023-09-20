#import pkg_resources
#pkg_resources.require("opencv-python==4.6.0.66")
import cv2 # Import the OpenCV library
import time
from pprint import *
import numpy as np
import _utils
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

### EX 3.1 ###

# See focal.py file in Ex3 folder

### EX 3.2 ###

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

    arlo.go_diff(_utils.leftWheelFactor*_utils.standardSpeed*0.6, _utils.rightWheelFactor*_utils.standardSpeed*0.6, r, l)
    
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
        arlo.go_diff(_utils.leftWheelFactor*_utils.standardSpeed, _utils.rightWheelFactor*_utils.standardSpeed, 0, 1)
        turnSeconds = _utils.degreeToSeconds(20)
        _utils.wait(turnSeconds)
        arlo.stop()
        _utils.wait(2.0)

        return False

def drive_to_landmarks(landmarks_lst):
    landmark = landmarks_lst[0]

    dist = landmark[0]
    angle = landmark[1]
    direction = landmark[2]
    id = landmark[3]

    print('Driving towards landmark ' + str(id))
    _utils.sharp_turn(direction, angle)
    driveSeconds = _utils.metersToSeconds(dist/1000)
    arlo.go_diff(_utils.leftWheelFactor*_utils.standardSpeed, _utils.rightWheelFactor*_utils.standardSpeed, 1, 1)
    _utils.wait(driveSeconds)

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
    

    #if np.dot(world_angle, np.array([1, 0, 0])) < 0:   # Hvis vinklen er negativ, så skal robotten dreje til højre
    #    arlo.go_diff(_utils.leftWheelFactor*_utils.standardSpeed, _utils.rightWheelFactor*_utils.standardSpeed, 1, 0)
    #    turnSeconds = _utils.degreeToSeconds(20)
    #    _utils.wait(turnSeconds)
    #    arlo.stop()
    #    _utils.wait(2.0)

    lst = []
    for i in range(len(ids)):
        world_angle = np.arccos(np.dot(tvecs[i]/np.linalg.norm(tvecs[i]), np.array([0, 0, 1])))
        print(world_angle)
        if np.dot(tvecs[i], np.array([1, 0, 0])) < 0:
            direction = 'right'
        else:
            direction = 'left'
    
        lst.append((linalg.norm(tvecs[i]), world_angle, direction, ids[i][0]))
    
    lst = lst.sort()
    print(lst)
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
            #drive_to_landmarks(landmarks_lst)
            time.sleep(15)      

camera()
