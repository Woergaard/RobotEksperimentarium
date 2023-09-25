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

        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        
    def landmarks(self, img):
        self.landmarks = _utils.pose_estimation(img, self.arucoDict)

        return self.landmarks

    def in_collision(self, pos):
        """
        find if the position is occupied or not. return if the queried pos is outside the map
        """
        indices = [int((pos[i] - self.map_area[0][i]) // self.resolution) for i in range(2)]
        for i, ind in enumerate(indices):
            if ind < 0 or ind >= self.n_grids[i]:
                return 1
        
        return self.grid[indices[0], indices[1]] 

    def populate(self, n_obs=6):
        """
        generate a grid map with some circle shaped obstacles
        """
        #origins = np.random.uniform(
        #    low=self.map_area[0] + self.map_size[0]*0.2, 
        #    high=self.map_area[0] + self.map_size[0]*0.8, 
        #    size=(n_obs, 2))
        #radius = np.random.uniform(low=0.1, high=0.3, size=n_obs)
        
        landmarks_lst = self.landmarks

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



# Arlo radius: 22.5 cm
# Boks radius (længeste side): 17.5 cm

##### SOFIE
    #radius_arlo = 22.5
    #radius_QR = 17.5 

def build_map(img, arucoDict):
    landmarks_lst = _utils.pose_estimation(img, arucoDict)

    plt.plot(0,0, 'bo')
    plt.annotate('ArloCinque', xy=(0, 0))
    
    for landmark in landmarks_lst:
        x = landmark[4][0]
        z = landmark[4][2]
        id = landmark[3]
        plt.Circle((x, z), 175)
        plt.annotate(str(id), xy=(x,z))
    
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()


class Landmark:
    def __init__(self, distance, vinkel, retning, id, tvec):
        self.distance = distance
        self.vinkel = vinkel
        self.retning = retning
        self.id = id
        self.tvec = tvec

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
            map = GridOccupancyMap()
            map.landmarks(image)
            map.populate()

            plt.clf()
            map.draw_map()
            plt.show()


camera2('thomas')
