import cv2 # Import the OpenCV library
import time
#from pprint import *
import numpy as np
import robot
from numpy import linalg
from time import sleep
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

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
            plt.plot(landmark.x, landmark.z, 'ro')
            plt.annotate(str(landmark.id), xy=(landmark.x, landmark.z))
    
    def draw_tree(self, G):
        #for node in G.nodes:
        #    plt.plot(node.x, node.z, 'go')
        self.nodes += G.nodes
        self.edges += G.edges
        
        for edge in G.edges:
            plt.plot([edge[0].x, edge[0].z] , [edge[1].x, edge[1].z], 'go-')
    
    def draw_goal(self, goal):
        plt.plot(goal.x, goal.z, 'yo')
        plt.annotate('Mål', xy=(goal.x, goal.z))

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

def is_spot_free(spot, landmarks):
    box_radius = 175.0

    for landmark in landmarks:        
        if dist(spot, landmark) < box_radius:
            print('Occupied by ' + str(landmark.id))
            return False

    #print('Spot free!')
    return True
'''
###############
# Caro Leger 
###############
        if animation:
            self.draw_tree(spot)
            if writer is not None:
                writer.grab_frame()
self, animation=True, writer=None
#####
'''
    

def find_nearest_node(x_new, G):
    distances = []
    for node in G.nodes:
        distances.append(abs(dist(x_new, node)))
    nearest_i = np.argmin(distances)
    return G.nodes[nearest_i], nearest_i


def make_edge(nearest_node, steering_node):
    x = steering_node.x - nearest_node.x
    z = steering_node.z - nearest_node.z
    edge = Node(x, z, nearest_node)
    print(edge)
    return edge
    


def steer(nearest_node, steering_node, stepLength):
    #edge = make_edge(nearest_node, steering_node)
    #nævner = np.sqrt(edge.x**2 + edge.z**2)
    #x = nearest_node.x - (edge.x*stepLength)/nævner
    #z = nearest_node.z - (edge.z*stepLength)/nævner

    #nævner = np.sqrt(steering_node.x**2 + steering_node.z**2)
    x = (nearest_node.x - steering_node.x) * stepLength / linalg.norm(nearest_node.x - steering_node.x)#nævner + steering_node.x
    z = (nearest_node.z - steering_node.z) * stepLength / linalg.norm(nearest_node.z - steering_node.z) #nævner + steering_node.z
    
    x = steering_node.x + x
    z = steering_node.z + z

    print("x:", x, "z:", z)
    
    new_node = Node(x, z, nearest_node)
    return new_node

### RRT ###
#  Path planning med Rapidly-exploring random trees
def RRT(goal, mapsizex, mapsizez, maxiter, landmarks, rootNode, stepLength):
    G = Graf([rootNode], [])
    iters = 0

    while iters < maxiter:
        steering_node = Node(random.randrange(-mapsizex, mapsizex), random.randrange(0, mapsizez), None)
        if is_spot_free(steering_node, landmarks):
            iters += 1
            print(iters)
            nearest_node, nearest_i = find_nearest_node(steering_node, G)
            new_node = steer(nearest_node, steering_node, stepLength)
            print("new node:", new_node.x, new_node.z, "steering node:", steering_node.x, steering_node.z)
            G.nodes.append(new_node)
            G.edges.append((nearest_node, new_node))

            if not is_spot_free(new_node, [goal]):
                return G, new_node
            
            # INDSÆT TJEK FOR, OM DEN NYE NODE HAR FRI PASSAGE TIL GOAL
    
    return G, new_node


### MAIN ###
def run_RRT(img, arucoDict, draw):
    landmarks = landmark_detection(img, arucoDict)
    for landmark in landmarks:
        print(landmark.id)

    goal = Landmark(None, None, None, 'mål', [0, 0, 3000])  # lav om evt. landmarks[-1]

    rootNode = Node(0, 0, None)
    stepLength = 200 # milimeter
    maxiter = 250

    ourMap = Map(4000, 4000)

    G, new_node = RRT(goal, ourMap.xlim, ourMap.zlim, maxiter, landmarks, rootNode, stepLength)

    if draw:
        ourMap.draw_tree(G)
        ourMap.draw_landmarks(landmarks)
        ourMap.draw_goal(goal)
        ourMap.show_map()
    

    '''
    ### animation begynder her


    show_animation = True
    metadata = dict(title="RRT Test")
    writer = FFMpegWriter(fps=15, metadata=metadata)
    fig = plt.figure()

    show_animation = True 
    
    
    fig = plt.figure()

    with writer.saving(fig, "writer_test.mp4", 100):
        path = run_RRT.is_spot_free(animation=show_animation, writer=writer)
    
        if path is None:
            print("Cannot find path")
        else:
            print("found path!!")

            # Draw final path
            if show_animation:
                ourMap.draw_tree()
                #plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
                plt.grid(True)
                #plt.pause(0.01)  # Need for Mac
                plt.show()
                writer.grab_frame()

########'''

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

            is_spot_free(574, 846, landmarks)
            sleep(2)

        if command == 'RRT':
            run_RRT(image, arucoDict, True)

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