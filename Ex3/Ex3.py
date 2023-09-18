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


def height_of_QR_in_image(img):
    '''
    Funktionen returnerer højden af en box på et billede i pixels.n
    Argumenter:
        image:  det givne billede
    '''
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    
    aruco_corners, _, _ = cv2.aruco.detectMarkers(img, arucoDict) 
    print(aruco_corners)


    # If at least one marker is detected
    if len(aruco_corners) > 0:
        for corner in aruco_corners:
            # The corners are ordered as top-left, top-right, bottom-right, bottom-left
            top_left = corner[0][0]
            bottom_left = corner[0][3]
            
            # Compute the height in pixels
            height = bottom_left[1] - top_left[1]
            print("Height of the ArUco marker:", height, "pixels")


def camera_setup():
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

        height_of_QR_in_image(image)
    
camera_setup()




### EX 3.1 ###

# See focal.py file in Ex3 folder

### EX 3.2 ###

