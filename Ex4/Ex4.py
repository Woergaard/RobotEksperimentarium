import cv2 # Import the OpenCV library
import time
#from pprint import *
import numpy as np
import robot
from numpy import linalg
from time import sleep
import random
import _utils
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import math

arlo = robot.Robot()

try:
    import picamera2
    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)

print("OpenCV version = " + cv2.__version__)




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
            plt.plot(landmark.x, landmark.z, 'ro', markersize = 17)
            plt.annotate(str(landmark.id), xy=(landmark.x, landmark.z))
    
    def draw_tree(self, G):
        #for node in G.nodes:
        #    plt.plot(node.x, node.z, 'go')
        nodes = G.nodes
        edges = G.edges
        
        #print('Kanter: ' + str(G.edges))
        #print('Noder: ' + str(G.nodes))
        for edge in edges:

            #print('Nearest node: ', edge[0].x, edge[0].z, 'New node: ', edge[1].x, edge[1].z)

            plt.plot([edge[0].x, edge[1].x] , [edge[0].z, edge[1].z], 'o-')
        
        for node in nodes: 
            plt.plot(node.x, node.z, 'go')
    
    def draw_goal(self, goal):
        plt.plot(goal.x, goal.z, 'yo', markersize = 17)
        plt.annotate('Mål', xy=(goal.x, goal.z))

    def draw_path(self, path):
        for node in path:
            plt.plot(node.x, node.z, 'mo')

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

def find_turn_angle(position, node):
    myradians = math.atan2(node.z-position.z, node.x-position.x)
    mydegrees = math.degrees(myradians)

    mydegrees = mydegrees - 90

    if mydegrees < 0:
        direction = 'left'
    else:
        direction = 'right'

    return direction, mydegrees

def inverse_direction(direction):
    if 'left':
        return 'right'
    elif 'right':
        return 'left'

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
    
    if ids == []: 
        print("No marker detected")
        return lst
    else:
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

def is_spot_free(spot, landmarks, radius):

    for landmark in landmarks:        
        if dist(spot, landmark) < radius:
            print('Occupied by ' + str(landmark.id))
            return False
    #print('Spot free!')
    return True
    

def find_nearest_node(x_new, G):
    distances = []
    for node in G.nodes:
        distances.append(abs(dist(x_new, node)))
    nearest_i = np.argmin(distances)
    print('nearest i:', nearest_i)
    return G.nodes[nearest_i]



def steer(nearest_node, steering_node, stepLength):
    # s er antal skridt, så vi itere over denne hver gang et skridt er taget for at tjekke om der er frit. 
    steering_vec = np.array([steering_node.x, steering_node.z])
    nearest_vec = np.array([nearest_node.x, nearest_node.z])

    v = steering_vec - nearest_vec
    e = v / linalg.norm(v)
    q = nearest_vec + stepLength * e
    
    new_node = Node(q[0], q[1], nearest_node)
    return new_node

### RRT ###
#  Path planning med Rapidly-exploring random trees
def RRT(goal, mapsizex, mapsizez, maxiter, landmarks, rootNode, stepLength):
    G = Graf([rootNode], [])
    iters = 0
    while iters < maxiter:
        steering_node = Node(random.randrange(-mapsizex, mapsizex), random.randrange(0, mapsizez), None)
        box_radius = 400.0

        if is_spot_free(steering_node, landmarks, box_radius):
            iters += 1
            print(iters)
            nearest_node = find_nearest_node(steering_node, G)
            new_node = steer(nearest_node, steering_node, stepLength)
            
            print("new node:", new_node.x, new_node.z, "steering node:", steering_node.x, steering_node.z)
            G.nodes.append(new_node)
            G.edges.append((nearest_node, new_node))

            goal_radius = 200.0
            if not is_spot_free(new_node, [goal], goal_radius):
                goalNode = Node(goal.x, goal.z, new_node)
                G.nodes.append(goalNode)
                G.edges.append((new_node, goal))
                return G, new_node
            
            # INDSÆT TJEK FOR, OM DEN NYE NODE HAR FRI PASSAGE TIL GOAL
    
    return G, new_node


### MAIN ###
def run_RRT(img, arucoDict, draw, drive):
    landmarks = landmark_detection(img, arucoDict)
    for landmark in landmarks:
        print(landmark.id)

    goal = Landmark(None, None, None, 'mål', [0, 0, 3000])  # lav om evt. landmarks[-1]

    rootNode = Node(0, 0, None)
    stepLength = 100.0 # milimeter
    maxiter = 1500

    ourMap = Map(1500, 4000)

    G, new_node = RRT(goal, ourMap.xlim, ourMap.zlim, maxiter, landmarks, rootNode, stepLength)

    if draw:
        ourMap.draw_tree(G)
        ourMap.draw_landmarks(landmarks)
        ourMap.draw_goal(goal)

        path = []
        goalNode = G.nodes[-1]
        path.append(goalNode.parent)
        ourMap.draw_path(path)
        print('driving towards', goalNode)

        notRoot = True

        while notRoot:
            if path[-1].parent != None:
                path.append(path[-1].parent)
            else:
                notRoot = False

        path.pop()
        path.reverse()
        print("PATH: ", path)

        ourMap.draw_path(path)
        ourMap.show_map()

    if drive:
        prevnode = Node(0, 0, None)
        
        print('path', path)
        for node in path:
            direction, degrees = find_turn_angle(prevnode, node)
            print(direction, degrees) 
            _utils.sharp_turn(direction, degrees)
            _utils.drive('forward', stepLength)
            _utils.sharp_turn(inverse_direction(direction), degrees)
            prevnode = node
        
        print('Arrived at goal!')
    

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
    #WIN_RF = "Example 1"
    #cv2.namedWindow(WIN_RF)
    #cv2.moveWindow(WIN_RF, 100, 100)
    
    while cv2.waitKey(4) == -1: # Wait for a key pressed event
        image = cam.capture_array("main")
        
        # Show frames
        #cv2.imshow(WIN_RF, image)

        if command == 'build_map':
            landmarks = landmark_detection(image, arucoDict)

            ourMap = Map(4000, 4000)
            ourMap.draw_landmarks(landmarks)

            is_spot_free(574, 846, landmarks, 400.0)
            sleep(2)

        if command == 'RRT':
            run_RRT(image, arucoDict, True, True)

camera('RRT')

