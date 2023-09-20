#import pkg_resources
#pkg_resources.require("opencv-python==4.6.0.66")
import cv2 # Import the OpenCV library
import time
from pprint import *
import numpy as np

try:
    import picamera2
    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)

print("OpenCV version = " + cv2.__version__)

import _utils
import robot

arlo = robot.Robot()

rightWheelFactor = 1.0
leftWheelFactor = 1.06225
standardSpeed = 31.0


def find_and_drive_to_landmark(img):
    '''
    Funktionen drejer om egen akse, indtil den har fundet et landmark OG der er frit. Derefter kører den mod landmarket.
    Argumenter:
        img:  billedet fra kameraet
    '''

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_corners, _, _ = cv2.aruco.detectMarkers(img, arucoDict)

    print(aruco_corners)
    
    if len(aruco_corners) > 0:
        for corner in aruco_corners:
            # The corners are ordered as top-left, top-right, bottom-right, bottom-left
            top_left = corner[0][0]
            top_right = corner[0][1]
            bottom_right = corner[0][2]
            bottom_left = corner[0][3]

            print('Landmark detected')
            print(top_left, top_right, bottom_right, bottom_left)

            if top_left[0] < 550 and top_left[0] > 350 and bottom_left[0] > 350 and bottom_left[0] < 550: # hvis landmark er ligeud
                arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)
            elif top_left[0] > 550 and bottom_left[0] > 550:
                arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 1) # drejer til venstre
            elif top_left[0] < 350 and bottom_left[0] < 350:
                arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 0) # drejer til højre
    else:
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 1)
        turnSeconds = _utils.degreeToSeconds(20)
        _utils.wait(turnSeconds)
        arlo.stop()
        _utils.wait(2.0)

def camera():
    # Open a camera device for capturing
    imageSize = (1280, 720)
    FPS = 30
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


    # Open a window
    WIN_RF = "Example 1"
    cv2.namedWindow(WIN_RF)
    cv2.moveWindow(WIN_RF, 100, 100)

    while cv2.waitKey(4) == -1: # Wait for a key pressed event
        image = cam.capture_array("main")
        
        # Show frames
        cv2.imshow(WIN_RF, image)

        find_and_drive_to_landmark(image)

camera()