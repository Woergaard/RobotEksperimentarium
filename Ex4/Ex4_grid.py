import cv2 # Import the OpenCV library
import time
import _utils
from pprint import *
import numpy as np
import robot
from numpy import linalg

import matplotlib.pyplot as plt

arlo = robot.Robot()

try:
    import picamera2
    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)

print("OpenCV version = " + cv2.__version__)


import numpy as np
import matplotlib.pyplot as plt

class GridOccupancyMap(object):
    """

    """
    def __init__(self, low=(0, 0), high=(5, 5), res=0.005) -> None:
        self.map_area = [low, high]    #a rectangular area    
        self.map_size = np.array([high[0]-low[0], high[1]-low[1]])
        self.resolution = res

        self.n_grids = [ int(s//res) for s in self.map_size]

        self.grid = np.zeros((self.n_grids[0], self.n_grids[1]), dtype=np.uint8)

        self.extent = [self.map_area[0][0], self.map_area[1][0], self.map_area[0][1], self.map_area[1][1]]


    def populate(self,landmarks_lst, n_obs=6):
        """
        generate a grid map with some circle shaped obstacles
        """

        radius_arlo = 0.225
        radius_QR = 0.175

        radius = [radius_arlo]
        origins = [[2.5, 0.0]]
        for i in range(len(landmarks_lst)):
            angle = landmarks_lst[i][1] 
            distance = landmarks_lst[i][2] 

            x = distance * np.cos(angle)
            y = distance * np.sin(angle)

            radius.append(radius_QR)
            origins.append([x, y])

        #fill the grids by checking if the grid centroid is in any of the circle
        for i in range(self.n_grids[0]):
            for j in range(self.n_grids[1]):
                centroid = np.array([self.map_area[0][0] + self.resolution * (i+0.5), 
                                     self.map_area[0][1] + self.resolution * (j+0.5)])
                for o, r in zip(origins, radius):
                    if np.linalg.norm(centroid - o) <= r:
                        self.grid[i, j] = 1
                        break

    
    def draw_map(self):
        #note the x-y axes difference between imshow and plot
        plt.imshow(self.grid.T, cmap="Greys", origin='lower', vmin=0, vmax=1, extent=self.extent, interpolation='none')


def pose_estimation(img, arucoDict): 
    """Funktionen returnerer en liste af placeringer i kameraets koordinatsystem samt id på de givne QR-koder"""
    """Denne funktion skal finde positionen af et landmark i forhold til robotten og udregne vinklen og afstanden til landmarket."""

    # 1. Udregne afstanden til QR code - DONE 
    # 2. Udregn vinkel til QR code 
    # 3. Dreje robotten, så den er vinkelret på QR code
    # 4. Køre fremad, indtil robotten er tæt nok på QR code

    aruco_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, arucoDict)
    print(aruco_corners)
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
        if command == 'thomas':
            landmarks_lst = pose_estimation(image, arucoDict)
            map = GridOccupancyMap()
            map.populate(landmarks_lst)

            plt.clf()
            map.draw_map()
            plt.show()


camera2('thomas')
