import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys
import _utils
import math
import random

# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = False # Whether or not we are running on the Arlo robot

def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot

if isRunningOnArlo():
    # XXX: You need to change this path to point to where your robot.py file is located
    sys.path.append("../")

try:
    #import Ex5.src.handout.python.robot as robot
    onRobot = True
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False

# Landmarks.
CRED, CGREEN, CBLUE, CCYAN, CYELLOW, CMAGENTA, CWHITE, CBLACK = _utils.bgr_colors()

# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
#landmarkIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#landmarks = {
#    1: (0.0, 0.0),  # Coordinates for landmark 1
#    2: (300.0, 0.0),  # Coordinates for landmark 2
#    3: (0.0, 0.0), 
#    4: (0.0, 0.0), 
#    5: (0.0, 0.0), 
#    6: (0.0, 0.0), 
#    7: (0.0, 0.0), 
#    8: (0.0, 0.0), 
#    9: (0.0, 0.0), 
#    10: (0.0, 0.0), 
#    11: (0.0, 0.0), 
#    12: (0.0, 0.0), 
#}

landmarkIDs = [11, 2]
landmarks = {
    11: (0.0, 0.0),  # Coordinates for landmark 1
    2: (300.0, 0.0)  # Coordinates for landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks

# Main program #
try:
    if showGUI:
        # Open windows
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)

        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)

    # Initialize particles
    num_particles = 1000
    particles = _utils.initialize_particles(num_particles)
    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

    # Driving parameters
    velocity = 0.0 # cm/sec
    angular_velocity = 0.0 # radians/sec

    # Initialize the robot (XXX: You do this)
    if onRobot: 
        #arlo = robot.Robot()
        print('hej')

    # Allocate space for world map
    world = np.zeros((1000,1000,3), dtype=np.uint8)

    # Draw map
    _utils.draw_world(est_pose, particles, world, landmarks, landmarkIDs, landmark_colors)

    print("Opening and initializing camera")
    if camera.isRunningOnArlo():
        cam = camera.Camera(0, 'arlo', useCaptureThread = True)
    else:
        cam = camera.Camera(0, 'macbookpro', useCaptureThread = True)

    Xlst = []
    iters = 0
    while True: # ændre hvis vi vil køre flere and iters < 10
        # Move the robot according to user input (only for testing)
        action = cv2.waitKey(1)
        if action == ord('q'): # Quit
            break
    
        if not isRunningOnArlo():
            if action == ord('w'): # Forward
                velocity += 4.0
            elif action == ord('x'): # Backwards
                velocity -= 4.0
            elif action == ord('s'): # Stop
                velocity = 0.0
                angular_velocity = 0.0
            elif action == ord('a'): # Left
                angular_velocity += 0.2
            elif action == ord('d'): # Right
                angular_velocity -= 0.2

        # Use motor controls to update particles
        # XXX: Make the robot drive
        # XXX: You do this
        if onRobot: 
            None#_utils.drive('forwards', 2)

        # Fetch next frame
        colour = cam.get_next_frame()
        
        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        
        
        Xlst.append(particles) # således at Xlst[iter] er lig de nuværende particles
        
        sigma_theta = 0.57
        sigma_d = 5.0
        particle.add_uncertainty(particles, sigma_d, sigma_theta)

        if not isinstance(objectIDs, type(None)):
            landmarks_lst = _utils.make_list_of_landmarks(objectIDs, dists, angles, landmarks)
            
            _utils.update_weights(sigma_d, sigma_theta, landmarks_lst, particles)    
                
            _utils.normalize_weights(particles)

            intervals = _utils.make_intervals(particles)

            particles = _utils.generate_new_particles(num_particles, particles, intervals)

            # Draw detected objects
            cam.draw_aruco_objects(colour)
        else:
            # No observation - reset weights to uniform distribution
            for p in particles:
                p.setWeight(1.0/num_particles)
    
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        #def path_planning(img, arucoDict, draw, drive_after_plan): 
        #    
        #    _utils.RRT((0.0, 1500.0), 4000.0, 4000.0, 100, landmarks, est_pose, 300.0, 50)
        #
        #    _utils.run_RRT(img, arucoDict, draw, drive_after_plan)

        if showGUI:
            # Draw map
            _utils.draw_world(est_pose, particles, world, landmarks, landmarkIDs, landmark_colors)

            # Show frame
            cv2.imshow(WIN_RF1, colour)

            # Show world
            cv2.imshow(WIN_World, world)

        iters += 1

finally: 
    # Make sure to clean up even if an exception occurred
    
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()
