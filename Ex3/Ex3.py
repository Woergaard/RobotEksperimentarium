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
    Funktionen returnerer højden af en box på et billede i pixels.
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

    pprint(cam.camera_configuration()) # Print the camera configuration in use

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

# x = height of QR code in image
# X = height of QR code in image irl
# Z = dist
# f = focal length
"""
def height_of_QR_in_image(img):
    '''
    Funktionen returnerer højden af en box på et billede i pixels.
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
        
    #arucoMarkerLength = 145  # andet ord for X
    #intrincsic_matrix = 
    #rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(aruco_corners, arucoMarkerLength, intrinsic_matrix, distortion_coeffs)
    
    # x = 
    # return x # højde på billedet
"""


def focal_length(xlst, X, Zlst):
    '''
    Funktionen returnerer højden af en box på et billede i pixels.
    Argumenter:
        x = height of QR code in image
        X = height of QR code in image irl
        Z = dist
    '''
    flst = []
    for i in range(len(xlst)):
        flst.append(xlst[i] * (X/Zlst[i])) 

    fmean = np.mean(flst)
    fstd = np.std(flst)
    return fmean, fstd 



Zlst = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0] #length of distances
X = 145.0 # height of QR code IRL

xlst = [] # height of QR code in image


