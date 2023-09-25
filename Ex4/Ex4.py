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


def populate(self, n_obs=6):
    """
    generate a grid map with some circle shaped obstacles
    """
    _utils.pose_estimation()  
         
        #low=self.map_area[0] + self.map_size[0]*0.2, 
        #high=self.map_area[0] + self.map_size[0]*0.8, 
        #size=(n_obs, 2))

    origins = np.random.uniform(
            low=self.map_area[0] + self.map_size[0]*0.2, 
            high=self.map_area[0] + self.map_size[0]*0.8, 
            size=(n_obs, 2))


    radius_arlo = 22.5
    radius_QR = 17.5 

    #fill the grids by checking if the grid centroid is in any of the circle
    for i in range(self.n_grids[0]):
        for j in range(self.n_grids[1]):
            centroid = np.array([self.map_area[0][0] + self.resolution * (i+0.5), 
                                    self.map_area[0][1] + self.resolution * (j+0.5)])
            for o, r in zip(origins, radius_arlo, radius_QR):
                if np.linalg.norm(centroid - o) <= r:
                    self.grid[i, j] = 1
                    break

def camera(command):
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
            build_map(image, arucoDict)

camera()