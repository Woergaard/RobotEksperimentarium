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

import matplotlib.pyplot as plt

#radius_arlo = 22.5
#radius_QR = 17.5 

class Landmark:
    def __init__(self, distance, vinkel, retning, id, tvec):
        self.distance = distance
        self.vinkel = vinkel
        self.retning = retning
        self.id = id
        self.tvec = tvec

def landmark_detection(img, arucoDict): 
    """Funktionen returnerer en liste af placeringer i kameraets koordinatsystem samt id på de givne QR-koder"""
    """Denne funktion skal finde positionen af et landmark i forhold til robotten og udregne vinklen og afstanden til landmarket."""

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
        if np.dot(tvecs[i], np.array([1, 0, 0])) < 0:
            direction = 'left'
        else:
            direction = 'right'

        lst.append(Landmark(linalg.norm(tvecs[i]), world_angle, direction, ids[i][0], tvecs[i][0]))
        print('dist:' + str(linalg.norm(tvecs[i])) + ', vinkel:' + str(world_angle) + direction + ', id:' + str(ids[i][0]) + ', tvec:' + str(tvecs[i][0]))
    
    lst.sort(key=lambda x: x.distance, reverse=False)

    return lst

def build_map(landmarks_lst):
    plt.plot(0,0, 'bo')
    plt.annotate('ArloCinque', xy=(0, 0))

    for landmark in landmarks_lst:
        x = landmark.tvec[0]
        z = landmark.tvec[2]
        plt.Circle((x, z), 175)
        plt.annotate(str(landmark.id), xy=(x,z))
    
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()

def is_spot_free(spotx, spotz, landmarks_lst):
    box_radius = 175.0

    for landmark in landmarks_lst:
        x = landmark.tvec[0]
        z = landmark.tvec[2]
        
        if np.sqrt((spotx-x)+(spotz-z)) < box_radius:
            print('Occupied by ' + str(landmark.id))
            return False

    return True


##### KØRSEL ####

def camera2(command):
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

        if command == 'build_map':
            landmarks_lst = landmark_detection(image, arucoDict)
            build_map(landmarks_lst)
            is_spot_free(574, 846, landmarks_lst)

camera2('build_map')