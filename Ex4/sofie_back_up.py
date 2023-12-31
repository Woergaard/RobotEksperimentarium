import cv2 # Import the OpenCV library
import time
#from pprint import *
import numpy as np
import robot
from numpy import linalg
from time import sleep
import random

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

### EXERCISE 1 ###

class Landmark:
    def __init__(self, distance, vinkel, retning, id, tvec):
        self.distance = distance
        self.vinkel = vinkel
        self.retning = retning
        self.id = id
        self.tvec = tvec
        self.x = tvec[0]
        self.z = tvec[2]

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
        print('dist: ' + str(linalg.norm(tvecs[i])) + ', vinkel: ' + str(world_angle) + direction + ', id: ' + str(ids[i][0]) + ', tvec: ' + str(tvecs[i][0]))
    
    lst.sort(key=lambda x: x.distance, reverse=False)

    return lst

# def build_map(landmarks):
#     plt.plot(0,0, 'bo')
#     plt.annotate('ArloCinque', xy=(0, 0))

#     for landmark in landmarks:
#         x = landmark.tvec[0]
#         z = landmark.tvec[2]
#         plt.plot(x,z, 'ro')
#         #plt.Circle((x, z), 175)
#         plt.annotate(str(landmark.id), xy=(x,z))
    
#     plt.ylim(0, 10000)
#     plt.xlim(-10000, 10000)
#     plt.xlabel('x')
#     plt.ylabel('z')
#     plt.show()

# def is_spot_free(spotx, spotz, landmarks):
#     box_radius = 175.0

#     for landmark in landmarks:        
#         if np.sqrt((spotx-landmark.x)+(spotz-landmark.z)) < box_radius:
#             print('Occupied by ' + str(landmark.id))
#             return False

#     print('Spot free!')
#     return True


##### KØRSEL ####

# def camera(command):
#     # Open a camera device for capturing
#     imageSize = (1280, 720)
#     FPS = 60
#     cam = picamera2.Picamera2()
#     frame_duration_limit = int(1/FPS * 1000000) # Microseconds
#     # Change configuration to set resolution, framerate
#     picam2_config = cam.create_video_configuration({"size": imageSize, "format": 'RGB888'},
#                                                                 controls={"FrameDurationLimits": (frame_duration_limit, frame_duration_limit)},
#                                                                 queue=False)
#     cam.configure(picam2_config) # Not really necessary
#     cam.start(show_preview=False)

#     print(cam.camera_configuration()) # Print the camera configuration in use

#     time.sleep(1)  # wait for camera to setup

#     arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
#     # Open a window
#     WIN_RF = "Example 1"
#     cv2.namedWindow(WIN_RF)
#     cv2.moveWindow(WIN_RF, 100, 100)
    
#     while cv2.waitKey(4) == -1: # Wait for a key pressed event
#         image = cam.capture_array("main")
        
#         # Show frames
#         cv2.imshow(WIN_RF, image)
#         #landmark_drive('left', image, arucoDict)

#         if command == 'build_map':
#             landmarks = landmark_detection(image, arucoDict)
#             build_map(landmarks)
#             is_spot_free(574, 846, landmarks)
#             sleep(2)

# camera('build_map')


### EXERCISE 2 ### (TING ER DOBBELT)

### KLASSER ###

class Node:
    def __init__(self, x, z, parent):
        self.x = x
        self.z = z
        self.path = []
        self.parent = parent

class Graf:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

class Map:
    def __init__(self, xlim, zlim):
        self.landmarks = []
        self.nodes = []
        self.edges = []
        self.xlim = xlim
        self.zlim = zlim

    def draw_landmarks(self, landmarks):
        self.landmarks += landmarks

        for landmark in self.landmarks:
            plt.plot(landmark.x, landmark.z, 'ro')
            plt.annotate(str(landmark.id), xy=(landmark.x, landmark.z))
    
    def draw_tree(self, G):
        #for node in G.nodes:
        #    plt.plot(node.x, node.z, 'go')
        self.nodes += G.nodes
        self.edges += G.edges
        
        for edge in G.edges:
            plt.plot([edge[0].x, edge[0].z] , [edge[1].x, edge[1].z], 'go-')
    
    def show_map(self):
        plt.plot(0,0, 'bo')
        plt.annotate('ArloCinque', xy=(0, 0))
        
        plt.ylim(0, self.zlim)
        plt.xlim(-self.xlim, self.xlim)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.show()


### HJÆLPEFUNKTIONER, RRT ###
def dist(node1, node2):
    return np.sqrt((node1.x-node2.x)**2+(node1.z-node2.z)**2) 

def is_spot_free(spot, landmarks):
    box_radius = 175.0

    for landmark in landmarks:        
        if dist(spot, landmark) < box_radius:
            print('Occupied by ' + str(landmark.id))
            return False

    print('Spot free!')
    return True

def find_nearest_node(x_new, G):
    distances = []
    for node in G.nodes:
        distances.append(abs(dist(x_new, node)))
    nearest_i = np.argmin(distances)
    return G.nodes[nearest_i], nearest_i


def find_edge(nearest_node, steering_node):
    x = steering_node.x - nearest_node.x
    z = steering_node.z - nearest_node.z
    edge = Node(x, z, nearest_node)
    return edge


def steer(nearest_node, steering_node, stepLength):
    edge = find_edge(nearest_node, steering_node)
    nævner = np.sqrt(edge.x**2 + edge.z**2)
    x = nearest_node.x + (edge.x*10)/nævner
    z = nearest_node.z + (edge.z*10)/nævner
    new_node = Node(x, z, nearest_node)
    return new_node

# Path planning med Rapidly-exploring random trees
def RRT(goal, mapsizex, mapsizez, maxiter, landmarks, rootNode, stepLength):
    G = Graf([rootNode], [])
    iters = 0

    while iters < maxiter:
        steering_node = Node(random.randrange(-mapsizex, mapsizex), random.randrange(0, mapsizez), None)
        if is_spot_free(steering_node, landmarks):
            nearest_node, nearest_i = find_nearest_node(steering_node, G)
            new_node = steer(nearest_node, steering_node, stepLength)
            G.nodes.append(new_node)
            G.edges.append((nearest_node, new_node))

            if not is_spot_free(new_node, [goal]):
                return G, new_node




from matplotlib.animation import FFMpegWriter


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

        if command == 'RRT':
            run_RRT(image, arucoDict)

camera('RRT')



def run_RRT(img, arucoDict): 
    landmarks = landmark_detection(img, arucoDict)
    goal = landmarks[0]

    rootNode = Node(0, 0, None)
    stepLength = 100 # milimeter
    maxiter = 500

    ourMap = Map(10000, 10000)

    G, new_node = RRT(goal, ourMap.xlim, ourMap.zlim, maxiter, landmarks, rootNode, stepLength)
    
    ourMap.draw_landmarks(landmarks)
    ourMap.draw_tree(G)
    ourMap.show_map()
    

    






    show_animation = True 
    
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='RRT')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    
    fig = plt.figure()

    with writer.saving(fig, "writer_test.mp4", 100):
        path = RRT.__animation(animation=show, writer=writer)
        
        writer.grab_frame()


if __name__ == '__main__':
    main()


'''
            # NÅET TIL
            steer
            stepLength = 




            g.nodes 


            Xnearest = Nearest(G(V,E),Xnew) #find nearest vertex
            Link = Chain(Xnew,Xnearest)
            G.append(Link)
            if Xnew in Qgoal:
                Return G
        


        iters += 1

    return rootNode


Xnew  = RandomPosition()
    if IsInObstacle(Xnew) == True:
        continue
'''


import cv2 # Import the OpenCV library
import time
#from pprint import *
import numpy as np
import robot
from numpy import linalg
from time import sleep
import random

arlo = robot.Robot()

try:
    import picamera2
    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)

print("OpenCV version = " + cv2.__version__)

import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegWriter


#radius_arlo = 22.5
#radius_QR = 17.5 


### KLASSER ###

class Landmark:
    def __init__(self, distance, vinkel, retning, id, tvec):
        self.distance = distance
        self.vinkel = vinkel
        self.retning = retning
        self.id = id
        self.tvec = tvec
        self.x = tvec[0]
        self.z = tvec[2]
        
class Node:
    def __init__(self, x, z, parent):
        self.x = x
        self.z = z
        self.path = []
        self.parent = parent

class Graf:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

class Map:
    def __init__(self, xlim, zlim):
        self.landmarks = []
        self.nodes = []
        self.edges = []
        self.xlim = xlim
        self.zlim = zlim

    def draw_landmarks(self, landmarks):
        self.landmarks += landmarks

        for landmark in self.landmarks:
            plt.plot(landmark.x, landmark.z, 'ro')
            plt.annotate(str(landmark.id), xy=(landmark.x, landmark.z))
    
    def draw_tree(self, G):
        #for node in G.nodes:
        #    plt.plot(node.x, node.z, 'go')
        self.nodes += G.nodes
        self.edges += G.edges
        
        for edge in G.edges:
            plt.plot([edge[0].x, edge[0].z] , [edge[1].x, edge[1].z], 'go-')
    
    def show_map(self):
        plt.plot(0,0, 'bo')
        plt.annotate('ArloCinque', xy=(0, 0))
        
        plt.ylim(0, self.zlim)
        plt.xlim(-self.xlim, self.xlim)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.show()


### HJÆLPEFUNKTIONER, RRT ###
def dist(node1, node2):
    return np.sqrt((node1.x-node2.x)**2+(node1.z-node2.z)**2) 

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
        print('dist: ' + str(linalg.norm(tvecs[i])) + ', vinkel: ' + str(world_angle) + direction + ', id: ' + str(ids[i][0]) + ', tvec: ' + str(tvecs[i][0]))
    
    lst.sort(key=lambda x: x.distance, reverse=False)

    return lst

def is_spot_free(spot, landmarks, self, animation=True, writer=None):
    box_radius = 175.0

    for landmark in landmarks:        
        if dist(spot, landmark) < box_radius:
            print('Occupied by ' + str(landmark.id))
            return False

###############
# Caro Leger 
###############
        if animation:
            self.draw_graph(spot)
            if writer is not None:
                writer.grab_frame()

#####
    print('Spot free!')
    return True

def find_nearest_node(x_new, G):
    distances = []
    for node in G.nodes:
        distances.append(abs(dist(x_new, node)))
    nearest_i = np.argmin(distances)
    return G.nodes[nearest_i], nearest_i


def find_edge(nearest_node, steering_node):
    x = steering_node.x - nearest_node.x
    z = steering_node.z - nearest_node.z
    edge = Node(x, z, nearest_node)
    return edge


def steer(nearest_node, steering_node, stepLength):
    edge = find_edge(nearest_node, steering_node)
    nævner = np.sqrt(edge.x**2 + edge.z**2)
    x = nearest_node.x + (edge.x*10)/nævner
    z = nearest_node.z + (edge.z*10)/nævner
    new_node = Node(x, z, nearest_node)
    return new_node

### HOVEDFUNKTION ###
# Path planning med Rapidly-exploring random trees
def RRT(goal, mapsizex, mapsizez, maxiter, landmarks, rootNode, stepLength):
    G = Graf([rootNode], [])
    iters = 0

    while iters < maxiter:
        steering_node = Node(random.randrange(-mapsizex, mapsizex), random.randrange(0, mapsizez), None)
        if is_spot_free(steering_node, landmarks):
            nearest_node, nearest_i = find_nearest_node(steering_node, G)
            new_node = steer(nearest_node, steering_node, stepLength)
            G.nodes.append(new_node)
            G.edges.append((nearest_node, new_node))

            if not is_spot_free(new_node, [goal]):
                return G, new_node


### MAIN ###
def run_RRT(img, arucoDict): 
    landmarks = landmark_detection(img, arucoDict)
    goal = landmarks[0]

    rootNode = Node(0, 0, None)
    stepLength = 100 # milimeter
    maxiter = 500

    ourMap = Map(10000, 10000)

    G, new_node = RRT(goal, ourMap.xlim, ourMap.zlim, maxiter, landmarks, rootNode, stepLength)
    
    ourMap.draw_landmarks(landmarks)
    ourMap.draw_tree(G)
    ourMap.show_map()
    

    
    ### animation begynder her

    show_animation = True 
    
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='RRT')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    
    fig = plt.figure()

    with writer.saving(fig, "writer_test.mp4", 100):
        path = RRT.__animation(animation=show, writer=writer)
        
        writer.grab_frame()
        
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
            landmarks = landmark_detection(image, arucoDict)

            ourMap = Map(10000, 10000)
            ourMap.draw_landmarks(landmarks)

            is_spot_free(574, 846, landmarks)
            sleep(2)

        if command == 'RRT':
            run_RRT(image, arucoDict)

camera('RRT')


'''
if __name__ == '__main__':
    main()


            # NÅET TIL
            steer
            stepLength = 




            g.nodes 


            Xnearest = Nearest(G(V,E),Xnew) #find nearest vertex
            Link = Chain(Xnew,Xnearest)
            G.append(Link)
            if Xnew in Qgoal:
                Return G
        


        iters += 1

    return rootNode


Xnew  = RandomPosition()
    if IsInObstacle(Xnew) == True:
        continue
'''