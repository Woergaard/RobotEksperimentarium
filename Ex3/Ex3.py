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
            print(top_left)

"""
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
"""



### EX 3.1 ###

# See focal.py file in Ex3 folder

### EX 3.2 ###

### DEFINEREDE PARAMETRE ###
import _utils
import robot

arlo = robot.Robot()

rightWheelFactor = 1.0
leftWheelFactor = 1.06225
standardSpeed = 50.0


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

    arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, r, l)
    
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

#            if top_left[0] < 550 and top_left[0] > 350 and bottom_left[0] > 350 and bottom_left[0] < 550: # hvis landmark er ligeud
#                arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)
#            if top_left[0] > 550 and bottom_left[0] > 550:
#                arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 1)


#def drive_to_landmark():
#    top_left, top_right, bottom_right, bottom_left = camera('turn and watch') # drejer, indtil den finder et landmark
#    _utils.drive_and_sense()
#
#    arlo.stop()
#    #turn_until_landmark_is_center



################################################
#       Caroline's bud på Pose_estimation      #
################################################
'''
Pose estimation is the task of determining the
camera's position and orientation in relation to
one or more landmarks. There are many
methods of positioning
- frame, img
'''
             
''' 
pose_estimation
    - ids, boxese id (1, 2, 3...,N)
    - corners, the corners of the box 
    - rejectedImgPoints, the marker candidates that have been rejected during the identification step
'''         

# VI SKAL BENYTTE VORES FOCAL LENGTH HER!

def pose_estimation(frame, img, arucoDict, distortion_coeffs, arucoMarkerLength, intrinsic_matrix): 
    """Denne funktion skal finde positionen af et landmark i forhold til robotten og udregne vinklen og afstanden til landmarket."""
    aruco_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, arucoDict, ids, rejectedImgPoints)


    # Draw the detectet markers, if there is at least 1 marker
#    if (ids.size() > 0) :
#        cv2.aruco.drawDetectedMarkers(img, ids, aruco_corners)
    
    # if there is at least 1 marker
    if len(aruco_corners) > 0 :
        cv2.aruco.drawDetectedMarkers(img, ids, aruco_corners)
    
        
        rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(aruco_corners, arucoMarkerLength, intrinsic_matrix, distortion_coeffs) 
        # Draw a square around the markers
        cv2.aruco.drawDetectedMarkers(frame, aruco_corners, ids) 

        # Draw Axis
        cv2.aruco.drawAxis(frame, intrinsic_matrix, distortion_coeffs, rvecs, tvecs, 0.01)  


    return frame 


def landmark_drive(direction, img, frame, arucoDict, distortion_coeffs, arucoMarkerLength, intrinsic_matrix):
    """
    Denne funktion skal køre mod et landmark, indtil det er lige foran robotten.
    Hvis den mister landmarket, skal den roterer, indtil den finder det igen.
    Den skal kontinuerligt opdatere sin position i forhold til landmarket og 
    dreje, indtil den kan køre lige i mod det.
    """
    turn_and_watch(direction, img) 

    while turn_and_watch(direction, img): 
        pose_estimation(frame, img, arucoDict, distortion_coeffs, arucoMarkerLength, intrinsic_matrix)

    landmark_drive(direction, img, frame, arucoDict, distortion_coeffs, arucoMarkerLength, intrinsic_matrix)




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
    
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_corners,ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, arucoDict)

    while cv2.waitKey(4) == -1: # Wait for a key pressed event
        image = cam.capture_array("main")
        
        # Show frames
        cv2.imshow(WIN_RF, image)
        landmark_drive('left', image, frame, arucoDict, distortion_coeffs, arucoMarkerLength, intrinsic_matrix)


        #if command == 'turn and watch':
        #    turn_and_watch('left', image)

camera()